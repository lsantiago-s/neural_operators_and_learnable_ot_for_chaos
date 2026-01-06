from glm import pos
import wandb
import torch
import numpy as np
from typing import Callable
from pipeline.visualization import plot_l63_full_double, plot_summary_stats_comparison, plot_error_evolution
from matplotlib import pyplot as plt
from pipeline.visualization.lorentz63_plots import plot_l63_full_double, plot_l63_full_single
from .get_experiment_info import get_dataset_config
from pipeline.testing.metrics import compute_trajectory_errors, compute_summary_errors
from torch.nn.utils.clip_grad import clip_grad_norm_

def optimize_ot(epoch: int, ot_delay: int, summary_step_freq: int) -> bool:
    optimize = (epoch > ot_delay) and (epoch % summary_step_freq == 0)
    return optimize

def increase_λ_ot(epoch: int, ot_delay: int, λ_ot: float, λ_ot_increase: float) -> float:
    if epoch > ot_delay and (epoch % 10 == 0):
        if λ_ot > 0.0:
            λ_ot += λ_ot_increase
    return λ_ot

def normalize(s: torch.Tensor, eps: float|None = None) -> torch.Tensor:
    if eps is None:
        eps = 1e-6 if s.dtype == torch.float32 else 1e-12
    
    mean = s.mean(dim=1, keepdim=True).detach() # s = [B, T, n]; n=3 as R^3.
    
    std = s.std(dim=1, keepdim=True, correction=0).detach().clamp_min(eps)
    
    s_normalized = (s - mean) / std
    out = s_normalized.clamp(-10.0, 10.0)

    return out

def rollout(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    rollout_steps: int|None = None,
) -> torch.Tensor:
    """Full rollout of the emulator.

    Args:
        emulator (torch.nn.Module): Emulator of the dynamics.
        u (torch.Tensor): Data tensor of shape (B, T, d).
        param (torch.Tensor): Parameters for the emulator (problem dependent).

    Returns:
        torch.Tensor: Rollout predictions of shape (B, T, d).
    """
    B, T, U = u.shape
    u_hat = torch.zeros((B, T, U), dtype=u.dtype, device=u.device)
    u_hat[:, 0, :] = u[:, 0, :]
    steps = T if rollout_steps is None else rollout_steps
    for t in range(1, steps):
        u_hat[:, t, :] = emulator(u_hat[:, t-1, :], param)

    return u_hat

def anchored_rollout_with_short_horizon(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    loss_fn: Callable,
    short_horizon: int # must be at least 2
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_size, total_time_steps, spatial_dim = u.shape
    ic_indices = torch.arange(0, total_time_steps, short_horizon, device=u.device)
    last_ic = ic_indices[-1].item()
    last_prediction = last_ic + short_horizon - 1
    add_final_step = (last_prediction < total_time_steps - 1)

    num_ics = len(ic_indices)
    u_ics = u[:, ic_indices]

    current_state = u_ics.reshape(batch_size * num_ics, spatial_dim)
    params_expanded = param.repeat_interleave(num_ics, dim=0)
    
    all_predictions = [u_ics]
    loss_predictions = []
    loss_targets = []

    for step in range(short_horizon - 1):
        next_state = emulator(current_state, params_expanded)
        next_state_reshaped = next_state.reshape(batch_size, num_ics, spatial_dim)
        all_predictions.append(next_state_reshaped)

        target_time_indices = ic_indices + step + 1

        valid_mask = target_time_indices < total_time_steps
        if valid_mask.any():
            valid_targets = target_time_indices[valid_mask]
            u_target = u[:, valid_targets]
            u_pred = next_state_reshaped[:, valid_mask, :]
            loss_predictions.append(u_pred)
            loss_targets.append(u_target)
        current_state = next_state

    if loss_predictions:
        all_preds = torch.cat(loss_predictions, dim=1)
        all_targets = torch.cat(loss_targets, dim=1)
        loss = loss_fn(all_targets, all_preds)
    else:
        loss = torch.tensor(0.0, device=u.device)

    predictions = torch.stack(all_predictions, dim=2)
    predictions = predictions.reshape(batch_size, -1, spatial_dim)

    if add_final_step:
        predictions = torch.cat(
            [predictions, u[:, total_time_steps-1:total_time_steps, :]], 
            dim=1
        )
    predictions = predictions[:, :total_time_steps, :]

    assert predictions.shape == u.shape
    return predictions, loss

def assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        bad = ~torch.isfinite(x)
        idx = bad.nonzero(as_tuple=False)
        # keep it readable: show first few offending indices
        idx_preview = idx[:10].tolist()
        stats = {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "min": float(torch.nan_to_num(x).min().item()),
            "max": float(torch.nan_to_num(x).max().item()),
            "mean": float(torch.nan_to_num(x).mean().item()),
        }
        raise FloatingPointError(
            f"[NON-FINITE] {name}\n"
            f"stats={stats}\n"
            f"first_bad_indices={idx_preview}"
        )


def train_step(
    model: torch.nn.Module,
    f_φ: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    Lp: Callable,
    ot_c_φ: Callable,
    model_optimizer: torch.optim.Optimizer,
    summary_optimizer: torch.optim.Optimizer,
    λ_ot: float,
    rollout_steps: int,
    use_ot: bool,
    step_f_φ: bool,
    clip_summary_grad_norm: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:

    assert_finite(u, "u (input)")
    assert_finite(param, "param (input)")
    
    model.train()
    if f_φ.is_learnable:
        f_φ.train()

    model_optimizer.zero_grad()
    
    u_hat_anchored, Lp_batch = anchored_rollout_with_short_horizon(
        emulator=model,
        u=u,
        param=param,
        loss_fn=Lp,
        short_horizon=rollout_steps
    )
    assert_finite(u_hat_anchored, "u_hat_anchored (emulator output)")


    # ------------- i) Emulator step ---------------
    
    if use_ot:
        f_φ.requires_grad_(False)
        s = f_φ(u).detach()
        u_hat_full_rollout = rollout(
            emulator=model, u=u, param=param, rollout_steps=rollout_steps
        )
        s_hat = f_φ(u_hat_full_rollout)

        assert_finite(s, "s = f_φ(u)")
        assert_finite(s_hat, "s_hat = f_φ(u_hat_full_rollout)")

        if s.dim() == 2:
            s = s.unsqueeze(1)
            s_hat = s_hat.unsqueeze(1)
            
        if torch.isnan(s).any():
            print(f"[Warning] NaN detected in true summaries")
            
        if torch.isnan(s_hat).any():
            print(f"[Warning] NaN detected in pred summaries")

        ot_c_φ_batch = ot_c_φ(normalize(s), normalize(s_hat)) # s: [B, T*d , n] 

        model_loss = Lp_batch + λ_ot * ot_c_φ_batch

    else:
        model_loss = Lp_batch
        ot_c_φ_batch = torch.tensor(0.0)


    model_loss.backward()
    model_optimizer.step()


    # --------- ii) Discriminator step -------------

    for p in f_φ.parameters():
        p.requires_grad = True
        # torch.nan_to_num_(p.data, nan=0.0, posinf=0.0, neginf=0.0)
        # p.data.clamp_(-0.01, 0.01) # This is dangerous

    if use_ot:
        if f_φ.is_learnable:
            summary_optimizer.zero_grad()
        s = f_φ(u)
        u_hat_full_rollout = rollout(
            emulator=model, u=u, param=param, rollout_steps=rollout_steps
        )
        s_hat = f_φ(u_hat_full_rollout.detach())
        ot_c_φ_batch = ot_c_φ(normalize(s), normalize(s_hat))
        summary_loss = -ot_c_φ_batch

        if step_f_φ and f_φ.is_learnable:
            summary_loss.backward()
            summary_optimizer.step()
    else:
        with torch.no_grad():
            print("No OT (summaries unused)")
            s = f_φ(u)
            s_hat = f_φ(u_hat_anchored.detach())
            ot_c_φ_batch = torch.tensor(0.0)

    return (
        u_hat_anchored,
        s,
        s_hat,
        model_loss.item(),
        Lp_batch.item(),
        ot_c_φ_batch.item()
    )

def log_trajectory_visualizations(u_sample: np.ndarray, 
                                  u_hat_sample: np.ndarray, 
                                  s_sample: np.ndarray, 
                                  s_hat_sample: np.ndarray, 
                                  epoch: int, 
                                  global_step: int,
                                  **kwargs):
    dataset_config = get_dataset_config(kwargs.get("train_data_path", {}).replace('npz', 'json'))

    traj_errors = compute_trajectory_errors(u_sample, u_hat_sample)
    summary_errors = compute_summary_errors(s_sample, s_hat_sample)

    wandb.log({
        'errors/trajectory_mse': traj_errors['mse'],
        'errors/trajectory_rmse': traj_errors['rmse'],
        'errors/trajectory_mae': traj_errors['mae'],
        'errors/trajectory_l2': traj_errors['l2_error'],
        'errors/trajectory_l1': traj_errors['l1_error'],
        'errors/trajectory_l_infinity': traj_errors['l_infinity_error'],
        'errors/trajectory_energy_spectrum': traj_errors['energy_spectrum'],
        'errors/trajectory_histogram_error': traj_errors['histogram_error'],
        'errors/trajectory_mse_x': traj_errors['mse_per_component'][0],
        'errors/trajectory_mse_y': traj_errors['mse_per_component'][1],
        'errors/trajectory_mse_z': traj_errors['mse_per_component'][2],
        'errors/trajectory_mae_x': traj_errors['mae_per_component'][0],
        'errors/trajectory_mae_y': traj_errors['mae_per_component'][1],
        'errors/trajectory_mae_z': traj_errors['mae_per_component'][2],
        'errors/trajectory_rmse_x': traj_errors['rmse_per_component'][0],
        'errors/trajectory_rmse_y': traj_errors['rmse_per_component'][1],
        'errors/trajectory_rmse_z': traj_errors['rmse_per_component'][2],
        'errors/summary_mse': summary_errors['mse'],
        'errors/summary_mae': summary_errors['mae'],
        'errors/summary_energy_spectrum': summary_errors['energy_spectrum'],
        'errors/summary_histogram_error': summary_errors['histogram_error'],
        'errors/summary_l1': summary_errors['l1_error'],
        'errors/summary_l2': summary_errors['l2_error'],
        'errors/summary_l_infinity': summary_errors['l_infinity_error'],
        'epoch': epoch,
        'step': global_step
    })

    u_true_np = u_sample[0]
    u_hat_np = u_hat_sample[0]
    s_true_np = s_sample[0]
    s_hat_np = s_hat_sample[0]

    fig_traj = plot_l63_full_single(
        trajectory=u_hat_np,
        plotted_variable=f"${{\\hat{{u}}}}(t)$",
    )
    fig_traj_comp = plot_l63_full_double(
        traj1=u_hat_np, traj2=u_true_np,
        first=[r'$x(t)$_pred', r'$y(t)$_pred', r'$z(t)$_pred'],
        second=[r'$x(t)$_true', r'$y(t)$_true', r'$z(t)$_true'],
        plotted_variable_1=f"${{\\hat{{u}}}}(t)$",
        plotted_variable_2=r'$u(t)$',
    )
    fig_summary_stats = plot_summary_stats_comparison(
        s_true=s_true_np, s_pred=s_hat_np,
        title=f"Summary Statistics Comparison (epoch {epoch})",
    )
    fig_error = plot_error_evolution(traj_errors['mse_per_time'],
                                      title=f"Error Evolution (Epoch {epoch})")

    wandb.log({
        'visualizations/trajectories': wandb.Image(fig_traj),
        'visualizations/traj_comparison': wandb.Image(fig_traj_comp),
        'visualizations/summary_stats': wandb.Image(fig_summary_stats),
        'visualizations/error_evolution': wandb.Image(fig_error),
        'batch/step': global_step,
        'batch/epoch': epoch,
    })
    plt.close(fig_traj)
    plt.close(fig_traj_comp)
    plt.close(fig_summary_stats)
    plt.close(fig_error)

 