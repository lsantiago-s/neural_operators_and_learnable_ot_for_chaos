This is the implementation for the paper TBD

@article{tbd,
  title={TBD},
  author={TBD},
  journal={TBD},
  year={2025}
}

### Prepare the training data
To generate the problem data in the `datagen` folder, modify the ```/configs/datagen/datagen_config.json``` file to define the dynamical system and run:
```
python /pipeline/datagen/generate_data.py
```

Example:

- ```dt = 0.01```, ```subsample_stride = 100``` ⇒ ```dt_eff = 1.0```

So:

- ```crop_length = 1000``` corresponds to ~1000 time units
- ```transient_cutoff = 5000``` corresponds to ~5000 time units removed

To use crops mode, make ```t_end``` much larger in order to have a long usable trajectory.

### Optimal transport experiment

To train the emulator with the optimal transport (OT) method for chosen data, modify the config files (summary, emulator, train, eval) in ```/configs``` and run:

```bash
bash experiments/new/srun.sh
```

### Evaluation
To just evaluate a trained model, modify ```pipeline/testing/evaluate.py``` to fetch the correct dataset and checkpoint path (In the ```if __name__ == '__main__'``` block) and from the repo folder, run ```python -m pipeline.testing.evaluate```
