import json
from pathlib import Path
from dataclasses import dataclass
from pipeline.testing.get_dataset_path import get_dataset_path


@dataclass
class EvaluateConfig:
    test_data_path: str
    checkpoint_path: str
    device: str
    dtype: str
    noise_level: float
    rollout_steps: int | None
    crop_window_size: int | None

    def post_init(self):
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        if self.rollout_steps is not None and self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if self.crop_window_size is not None and self.crop_window_size <= 0:
            raise ValueError("crop_window_size must be positive or None")

def get_eval_configs(config_path: Path, exp_path: Path) -> dict[str, EvaluateConfig]:
    with open(config_path, 'r') as f:
        eval_config_dict = json.load(f)
        eval_config_dict['checkpoint_path'] = exp_path / "checkpoints" / "best_model.pth"
        eval_config_dict['test_data_path'] = get_dataset_path(str(exp_path))
        eval_config_obj = EvaluateConfig(**eval_config_dict)
        eval_config_obj.post_init()

    eval_config = {
        "eval_config": eval_config_obj,
    }
    return eval_config