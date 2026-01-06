import logging
import numpy as np
from typing import Any
from pathlib import Path
from scipy.integrate import solve_ivp
from collections.abc import Callable
from pipeline.datagen.config import DataGenConfig
from pipeline.datagen.dynamical_systems import IVP_MAP
from pipeline.datagen.postprocessing import crop_ok_lorenz63_two_lobes

logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, config: DataGenConfig | str | dict):
        if isinstance(config, (str, Path)):
            config = DataGenConfig.from_json(config)
        elif isinstance(config, dict):
            config = DataGenConfig(**config)

        self.config = config
        self.rng = np.random.default_rng()
        self.ode_func: Callable = IVP_MAP[config.experiment]
    def sample_parameters(self) -> np.ndarray:
        params = []
        for _, (low, high) in self.config.param_ranges.items():
            sampled = self.rng.uniform(low, high)
            params.append(sampled)
        return np.stack(params, axis=-1)

    def _solve_ivp(self, params: np.ndarray):
        logger.info("solving IVP with params: %s", params)
        logger.info("t_start: %f, t_end: %f, dt: %f", self.config.t_start, self.config.t_end, self.config.dt)
        logger.info("Total time steps: %d", int((self.config.t_end - self.config.t_start) / self.config.dt))
        t_span = (self.config.t_start, self.config.t_end)
        t_eval = np.arange(self.config.t_start, self.config.t_end, self.config.dt)
        init_conditions = self.rng.normal(0, 1, self.config.n_dim)

        solution = solve_ivp(
            fun=self.ode_func,
            t_span=t_span,
            y0=init_conditions,
            args=tuple(params),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )

        step = self.config.subsample_stride
        logger.info("Subsampling trajectory with stride: %d", step)
        trajectory = solution.y.T[::step, :]
        logger.info("Generated trajectory shape: %s", trajectory.shape)
        return trajectory
    
    def generate_one(self, idx: int) -> tuple[str, np.ndarray, np.ndarray]:
        params = self.sample_parameters()
        traj = self._solve_ivp(params)
        traj_id = f"{idx:06d}"
        return traj_id, traj, params

    def generate_dataset(self, progress: bool=True):
        if self.config.sampling_mode == "independent":
            return self._generate_independent(progress)
        return self._generate_crops(progress)

    def _generate_independent(self, progress: bool = True):
        iterator = range(self.config.n_samples)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Generating trajectories")

        return [self.generate_one(i) for i in iterator]

    def _generate_crops(self, progress: bool = True):
        params = self.sample_parameters()
        full_traj = self._solve_ivp(params)  # (T, D)

        logger.info("Extracting crops from full trajectory of shape: %s", full_traj.shape)

        start_idx = int(self.config.transient_cutoff or 0)
        usable = full_traj[start_idx:, :]

        crop_len = int(self.config.crop_length or 0)
        max_start = usable.shape[0] - crop_len
        if max_start <= 0:
            raise ValueError("Not enough usable timesteps for cropping")

        v = self.config.crop_validator
        want_validate = v is not None and str(v.get("type", "")).strip() != ""

        results: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
        iterator = range(self.config.n_samples)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Extracting crops")

        tries = 0
        max_tries = 50 * self.config.n_samples  # safety cap

        i = 0
        while i < self.config.n_samples:
            if tries > max_tries:
                raise ValueError("Failed to find enough valid crops. Relax validator or increase t_end/crop_length.")
            tries += 1

            s = int(self.rng.integers(0, max_start + 1))
            crop = usable[s : s + crop_len, :]

            ok = True
            meta: dict[str, Any] = {
                "type": "crop",
                "start_idx": s + start_idx,
                "end_idx": s + start_idx + crop_len,
                "burn_in": start_idx,
                "crop_len": crop_len,
            }

            if want_validate:
                if v is not None and v["type"] == "lorenz63_two_lobes":
                    ok = crop_ok_lorenz63_two_lobes(
                        crop,
                        axis=int(v.get("axis", 0)),
                        min_switches=int(v.get("min_switches", 2)),
                        min_lobe_fraction=float(v.get("min_lobe_fraction", 0.15)),
                    )
                else:
                    raise ValueError(f"Unknown crop_validator.type: {v['type']}")
                meta["validator"] = v["type"]

            if not ok:
                continue

            traj_id = f"{i:06d}"
            results.append((traj_id, crop, params, meta))
            i += 1

        return results