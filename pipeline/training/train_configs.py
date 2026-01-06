import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

@dataclass
class TrainConfig:
    #------------------------
    # Experiment settings
    train_data_path: str
    val_data_path: str
    device: str
    dtype: str
    epochs: int
    batch_size: int
    ot_delay: int 
    rollout_steps: int
    noise_level: float
    crop_window_size: int    
    summary_step_freq: int
    clip_summary_grad_norm: float
    #------------------------
    # Model settings
    emulator_lr: float
    emulator_config: dict[str, Any]
    emulator_optimizer_type: str
    #------------------------
    # Summary settings
    summary_config: dict[str, Any]
    summary_optimizer_type: str
    summary_lr: float
    #------------------------
    # Loss settings
    distribution_loss: str
    short_term_loss: str
    ot_penalty: float
    ot_penalty_increase: float
    feature_penalty: float
    geom_loss_p: int
    blur: float
    p_norm: float
    #------------------------

    def post_init(self):
        pass

def _filter_to_dataclass_keys(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(TrainConfig)}
    return {k: v for k, v in raw.items() if k in allowed}

def get_train_configs(config_paths: list[Path]) -> dict[str, TrainConfig]:
    with config_paths[0].open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    cfg: dict[str, Any] = dict(raw)
    nested = cfg.pop("train_config", None)
    if isinstance(nested, dict):
        cfg.update(nested)  # nested overrides win

    cfg = _filter_to_dataclass_keys(cfg)

    missing = [f.name for f in fields(TrainConfig) if f.name not in cfg]
    if missing:
        raise ValueError(
            f"Config missing required keys: {missing}\n"
            f"Config file: {config_paths[0]}"
        )

    train_config_obj = TrainConfig(**cfg)
    return {"exp_config": train_config_obj}