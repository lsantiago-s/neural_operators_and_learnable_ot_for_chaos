from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

def save_npz(
    data: list[
        tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]
        | tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]
    ],
    config: Any,
    output_path: str | Path,
) -> None:
    """
    Save a dataset in a flat npz structure:

      - ids: (N,) array of string ids
      - config: pickled DataGenConfig
      - traj_<id>: (T, D)
      - params_<id>: (P,)
      - meta_<id>: pickled dict (optional but always written as {} if missing)

    This remains compatible with existing training code that only reads ids/traj/params.
    """
    output_path = Path(output_path)

    ids: list[str] = []
    save_dict: dict[str, Any] = {
        "config": np.void(pickle.dumps(config)),
        "schema_version": "2",  # optional but useful for debugging
    }

    for item in data:
        if len(item) == 3:
            traj_id, traj, params = item
            meta: dict[str, Any] = {}
        elif len(item) == 4:
            traj_id, traj, params, meta = item
        else:
            raise ValueError("Each data item must be (id, traj, params) or (id, traj, params, meta).")

        traj_id = str(traj_id)
        ids.append(traj_id)

        save_dict[f"traj_{traj_id}"] = traj
        save_dict[f"params_{traj_id}"] = params
        save_dict[f"meta_{traj_id}"] = np.void(pickle.dumps(meta))

    save_dict["ids"] = np.asarray(ids, dtype=str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **save_dict)
    print(f"Saved {len(ids)} samples -> {output_path}")

    # Metadata JSON (human-readable)
    metadata = {
        "experiment": getattr(config, "experiment", None),
        "sampling_mode": getattr(config, "sampling_mode", "independent"),
        "n_saved_samples": len(ids),
        "n_base_samples": getattr(config, "n_samples", None),  # base solves (or crops count in crops mode)
        "t_start": getattr(config, "t_start", None),
        "t_end": getattr(config, "t_end", None),
        "subsample_stride": getattr(config, "subsample_stride", None),
        "dt": getattr(config, "dt", None),
        "dt_eff": (
            getattr(config, "dt", 0.0) * getattr(config, "subsample_stride", 1)
            if hasattr(config, "dt") and hasattr(config, "subsample_stride")
            else None
        ),
        "n_timesteps": getattr(config, "n_timesteps", None),
        "param_ranges": getattr(config, "param_ranges", None),
        "n_dim": getattr(config, "n_dim", None),
        "crop_length": getattr(config, "crop_length", None),
        "transient_cutoff": getattr(config, "transient_cutoff", None),
        "crop_validator": getattr(config, "crop_validator", None),
        "timestamp": datetime.now().isoformat(),
        "schema_version": "2",
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=4))
    print(f"Metadata -> {json_path}")
