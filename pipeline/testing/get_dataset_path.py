import torch
from pathlib import Path
from typing import Any

def _pick_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    
    preferred = [
        "best_model.pth",
    ]

    for name in preferred:
        p = ckpt_dir / name
        if p.is_file():
            return p
        
    candidates = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pth", ".pt"}:
            continue

    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]
    
    any_ckpts = [p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pt", ".pth"}]
    if any_ckpts:
        return max(any_ckpts, key=lambda p: p.stat().st_mtime) 
    
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}.")


def get_dataset_path(exp_path: str):
    ckpt_dir = Path(exp_path) / "checkpoints"
    ckpt_path = _pick_checkpoint(ckpt_dir)

    checkpoint: dict[str, Any] = torch.load(ckpt_path, map_location='cpu')

    info = checkpoint['dataset_info']['name'].split('_')
    dataset_path = f"data/{info[0]}/{info[1]}/test_data.npz"

    return dataset_path
