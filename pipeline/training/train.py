import torch
import wandb
import logging
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pipeline.dataloader import dynamical_system_dataset as ds, transforms as tf
from models.operator import initialize_operator
from models.optimization import initialize_optimizer
from models.optimization import initialize_loss_function
from models.summary import initialize_summary_stats
from .runtime_resolve import resolve_runtime_config
from .train_loop_helper import train_step, optimize_ot, log_trajectory_visualizations, increase_λ_ot
from pipeline.testing.eval_rollout import eval_rollout
from .get_experiment_info import get_problem_name, get_exp_path, get_exp_name, get_timestamp

logger = logging.getLogger(__file__)

def train(train_config: dict, use_wandb: bool):
    exp_cfg = train_config['exp_config']
    problem_name = get_problem_name(exp_cfg.train_data_path)
    ot_type = exp_cfg.summary_config['type'] if exp_cfg.ot_penalty > 0 else 'no_ot'
    clean_or_noisy = 'noisy' if exp_cfg.noise_level > 0 else 'clean'
    dataset_hash = exp_cfg.train_data_path.split('/')[-2]
    dataset_name = problem_name + '_' + dataset_hash
    exp_name = get_exp_name(problem_name, ot_type, clean_or_noisy)
    timestamp = get_timestamp()
    output_path = get_exp_path( exp_name, timestamp)

    device = torch.device(exp_cfg.device)
    dtype = getattr(torch, exp_cfg.dtype)
    config = {
                'experiment': str(output_path),
                'device': exp_cfg.device,
                'dtype': exp_cfg.dtype,
                'epochs': exp_cfg.epochs,
                'batch_size': exp_cfg.batch_size,
                'emulator_lr': exp_cfg.emulator_lr,
                'emulator_config': exp_cfg.emulator_config,
                'emulator_optimizer_type': exp_cfg.emulator_optimizer_type,
                'summary_type': exp_cfg.summary_config['type'],
                'summary_config': exp_cfg.summary_config,
                'summary_optimizer_type': exp_cfg.summary_optimizer_type,
                'summary_lr': exp_cfg.summary_lr,
                'short_term_loss': exp_cfg.short_term_loss,
                'distribution_loss': exp_cfg.distribution_loss,
                'ot_penalty': exp_cfg.ot_penalty,
                'noise_level': exp_cfg.noise_level,
                'rollout_steps': exp_cfg.rollout_steps,
                'crop_window_size': exp_cfg.crop_window_size,
                'summary_step_freq': exp_cfg.summary_step_freq,
                'dataset': dataset_name,
                'timestamp': timestamp
            }

    if use_wandb: 
        wandb.init(
            project="chaos-emulator",
            name=f"{exp_name}",
            config=config
        )

    train_dataset = ds.DynamicalSystemDataset(
        npz_path=exp_cfg.train_data_path,
        device=device,
        dtype=dtype,
        transforms=[
            # tf.NormalizeTransform(stats_path="data/lorenz63/66f92019/normalization_stats.npy"),
            tf.AddNoiseTransform(noise_level=exp_cfg.noise_level),
            tf.CropTransform(length=exp_cfg.crop_window_size)
        ]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=exp_cfg.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cpu')
    )

    val_dataset = ds.DynamicalSystemDataset(
        npz_path=exp_cfg.val_data_path,
        device=device,
        dtype=dtype,
        transforms=[
            # tf.NormalizeTransform(stats_path="data/lorenz63/66f92019/normalization_stats.npy"),
            tf.AddNoiseTransform(noise_level=exp_cfg.noise_level),
            tf.CropTransform(length=exp_cfg.crop_window_size)
        ]
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=exp_cfg.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cpu')
    )

    resolve_runtime_config(exp_cfg)

    model = initialize_operator(operator_config=exp_cfg.emulator_config, device=device, dtype=dtype)
    Lp = initialize_loss_function(loss_config=exp_cfg.short_term_loss)
    ot_c_φ = initialize_loss_function(loss_config=exp_cfg.distribution_loss)
    λ_ot = exp_cfg.ot_penalty
    model_optimizer = initialize_optimizer(model=model, learning_rate=exp_cfg.emulator_lr, optimizer_type=exp_cfg.emulator_optimizer_type)

    f_φ = initialize_summary_stats(summary_config=exp_cfg.summary_config, device=device, dtype=dtype)

    if f_φ.is_learnable:
        summary_optimizer = initialize_optimizer(model=f_φ, learning_rate=exp_cfg.summary_lr, optimizer_type=exp_cfg.summary_optimizer_type)
    else:
        summary_optimizer = None

    if use_wandb:
        wandb.config.update({'model_architecture': str(model)})
        wandb.config.update({'summary_architecture': str(f_φ)})
        wandb.watch(model, log='all', log_freq=100)
        wandb.watch(f_φ, log='all', log_freq=100)

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(exp_cfg.epochs):
        λ_ot = increase_λ_ot(epoch=epoch, ot_delay=exp_cfg.ot_delay, λ_ot=λ_ot, λ_ot_increase=exp_cfg.ot_penalty_increase)
        
        epoch_metrics = {
            'train/λ_ot': λ_ot,
            'train/epoch': epoch,
            'train/train_total_loss': 0.0,
            'train/train_lp_loss': 0.0,
            'train/train_ot_loss': 0.0,
            'val/epoch': epoch,
            'val/val_total_loss': 0.0,
            'val/val_lp_loss': 0.0,
            'val/val_ot_loss': 0.0,
        }
        
        num_batches = 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{exp_cfg.epochs}')

        use_ot = (epoch > exp_cfg.ot_delay) and (λ_ot > 0)
        step_f_φ = optimize_ot(epoch=epoch, ot_delay=exp_cfg.ot_delay, summary_step_freq=exp_cfg.summary_step_freq)

        for batch_idx, (_, u, param) in enumerate(pbar):
            train_u_hat, train_s, train_s_hat, train_total_loss, train_lp_loss, train_ot_c_φ_loss = train_step(
                model=model,
                f_φ=f_φ,
                u=u,
                param=param,
                Lp=Lp,
                ot_c_φ=ot_c_φ,
                model_optimizer=model_optimizer,
                summary_optimizer=summary_optimizer,
                λ_ot=λ_ot,
                rollout_steps=exp_cfg.rollout_steps,
                use_ot=use_ot,
                step_f_φ=step_f_φ,
                clip_summary_grad_norm=exp_cfg.clip_summary_grad_norm
            )

            epoch_metrics['train/train_total_loss'] += train_total_loss
            epoch_metrics['train/train_lp_loss'] += train_lp_loss
            epoch_metrics['train/train_ot_loss'] += train_ot_c_φ_loss
            num_batches += 1

            if use_wandb and global_step % 10 == 0:
                wandb.log({
                    'batch/total_loss': train_total_loss,
                    'batch/lp_loss': train_lp_loss,
                    'batch/ot_loss': train_ot_c_φ_loss,
                    'batch/step': global_step,
                    'batch/epoch': epoch,
                })

            if use_wandb and global_step % 10 == 0:
                u_sample = train_u_hat.detach().cpu().numpy()
                u_hat_sample = train_u_hat.detach().cpu().numpy()
                s_sample = train_s.detach().cpu().numpy()
                s_hat_sample = train_s_hat.detach().cpu().numpy()
                from pipeline.visualization.lorentz63_plots import (plot_summary_stats_comparison)
                fig_summary_stats = plot_summary_stats_comparison(
                        s_true=s_sample[0], s_pred=s_hat_sample[0],
                        title=f"Summary Statistics Comparison (epoch {epoch})",
                    )
                log_trajectory_visualizations(
                    u_sample=u_sample, 
                    u_hat_sample=u_hat_sample, 
                    s_sample=s_sample, 
                    s_hat_sample=s_hat_sample, 
                    epoch=epoch, 
                    global_step=global_step,
                    train_data_path=exp_cfg.train_data_path,
                )

            pbar.set_postfix({
                'loss': f'{train_total_loss:.4f}',
                'Lp': f'{train_lp_loss:.4f}',
                'OT': f'{train_ot_c_φ_loss:.4f}'
            })

            global_step += 1

        model.eval()
        with torch.no_grad():
            val_losses = []
            for _, u_val, param_val in val_dataloader:
                u_hat_val = eval_rollout(model, u_val, param_val, rollout_steps=exp_cfg.rollout_steps)
                val_loss = Lp(u_val, u_hat_val).item()
                val_losses.append(val_loss)
                avg_val_loss = sum(val_losses) / len(val_losses)

     
        for key in ['train/train_total_loss', 'train/train_lp_loss', 'train/train_ot_loss']:
            epoch_metrics[key] /= num_batches

        if use_wandb:
            wandb.log(epoch_metrics)

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'emulator_config': exp_cfg.emulator_config,
                'summary_config': exp_cfg.summary_config,
                'model_state_dict': model.state_dict(),
                'summary_state_dict': f_φ.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'summary_optimizer_state_dict': summary_optimizer.state_dict() if f_φ.is_learnable else None,
                'train_loss': epoch_metrics['train/train_total_loss'],
                'train_config': config,
                'dataset_info': {
                    'name': dataset_name
                }
            }


            if f_φ.is_learnable:
                checkpoint['summary_state_dict'] = f_φ.state_dict()
                checkpoint['summary_optimizer_state_dict'] = summary_optimizer.state_dict()
            
            checkpoint_path = output_path / 'checkpoints' / f'_epoch_{epoch+1}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(checkpoint, checkpoint_path)
            json.dump(config, open(output_path / 'config.json', 'w'), indent=4)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint = {
                    'epoch': epoch,
                    'emulator_config': exp_cfg.emulator_config,
                    'summary_config': exp_cfg.summary_config,
                    'model_state_dict': model.state_dict(),
                    'summary_state_dict': f_φ.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict(),
                    'summary_optimizer_state_dict': summary_optimizer.state_dict() if f_φ.is_learnable else None,
                    'train_loss': epoch_metrics['train/train_total_loss'],
                    'train_config': config,
                    'dataset_info': {
                        'name': dataset_name
                    }
                }
                torch.save(best_checkpoint, checkpoint_path.parent / 'best_model.pth')
            
            if use_wandb:
                wandb.save(checkpoint_path)

    if use_wandb:
        wandb.finish()
    
    logger.info(f"checkpoint -> {checkpoint_path}")
    logger.info("Training complete!")

    return output_path