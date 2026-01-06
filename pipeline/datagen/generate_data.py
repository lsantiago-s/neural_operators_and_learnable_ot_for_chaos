import argparse
import hashlib
import json
import base64
import logging
from pathlib import Path
from pipeline.datagen.trajectory_generator import DataGenerator
from pipeline.datagen.config import DataGenConfig
from pipeline.datagen.save import save_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data(): ## RUN FROM REPO ROOT
    parser = argparse.ArgumentParser(description="Generate dynamical system dataset")
    parser.add_argument('--config', type=Path, default='./configs/datagen/datagen_config.json', help='Path to config JSON')
    args = parser.parse_args()

    base_config = DataGenConfig.from_json(args.config)

    exp_version = generate_version_hash(base_config.__dict__)

    for split in ['train', 'val', 'test']:
        split_config = base_config.__dict__.copy()
        if split == 'train':
            split_config['n_samples'] = base_config.n_samples
            split_config['output_dir'] = Path(base_config.output_dir) /base_config.experiment / exp_version / 'train_data.npz'
        elif split == 'val':
            split_config['n_samples'] = base_config.n_samples
            split_config['output_dir'] = Path(base_config.output_dir) /base_config.experiment / exp_version / 'val_data.npz'
        else: # test
            split_config['n_samples'] = base_config.n_samples
            split_config['output_dir'] = Path(base_config.output_dir) /base_config.experiment / exp_version / 'test_data.npz'

        split_data_gen_config = DataGenConfig(**split_config)
        gen = DataGenerator(split_data_gen_config)
        data = gen.generate_dataset(progress=True)

        save_npz(data, split_data_gen_config, split_config['output_dir'])

def generate_version_hash(config: dict) -> str:
    hash_obj = hashlib.sha256()
    json_str = json.dumps(config, sort_keys=True)
    base64_str = base64.b64encode(json_str.encode()).decode()
    hash_obj.update(base64_str.encode())
    return hash_obj.hexdigest()[:8]

if __name__ == "__main__":
    generate_data()