import argparse
from utils.trainer import train
import os
import torch
import numpy as np
import yaml

def get_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train model with given parameters")
    # Add arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
    parser.add_argument("--log_name", default="Fuse_ff_mamba_pointnext_b_Unet_10")
    return parser.parse_args()
    
def main():
    # Parse arguments
    args = parse_args()
    config = get_config(args.config)
    config['log_name'] = args.log_name
    config['save_dir'] = os.path.join(os.getcwd(), 'tl_logs')
    config["data_dir"] = (
        args.data_dir
        if args.data_dir is not None
        else os.path.join(os.getcwd(), "data")
    )
    
    # Multi-head logic
    train_on = config.get('train_on', ['rmf', 'ovf'])
    test_on = config.get('test_on', 'rmf')
    all_datasets = list(set(train_on + [test_on]))

    class_weights_map = {
        "rmf": [0.13429631, 0.02357711, 0.05467328, 0.04353036, 0.02462899, 0.03230562, 0.2605792, 0.00621396, 0.42019516],
        "ovf": [0.121, 0.033, 0.045, 0.090, 0.012, 0.041, 0.020, 0.103, 0.334, 0.010, 0.191]
    }
    class_names = {
        "rmf": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
        "ovf": ['AB', 'PO', 'MR', 'BF', 'CE', 'PW', 'MH', 'BW', 'SW', 'OR', 'PR']
    }
    season_map = {
        "rmf": ["img_s2_spring", "img_s2_summer", "img_s2_fall", "img_s2_winter"],
        "ovf": ["img_s2_2020_spring", "img_s2_2020_summer", "img_s2_2020_fall", "img_s2_2020_winter"]
    }

    config['class_weights_map'] = {k: torch.tensor(v).float() for k, v in class_weights_map.items() if k in all_datasets}
    config['class_names'] = {k: v for k, v in class_names.items() if k in all_datasets}
    config['n_classes'] = {k: len(v) for k, v in class_names.items() if k in all_datasets}
    config['seasons_map'] = {k: season_map[k] for k in all_datasets}

    # Set test-specific configs
    config['test_classes'] = config['class_names'][test_on]
    config['test_n_classes'] = config['n_classes'][test_on]
    config['test_class_weights'] = config['class_weights_map'][test_on]
    config['test_seasons'] = config['seasons_map'][test_on]
    
    os.makedirs(config['save_dir'], exist_ok=True)
    print(config)

    # Call the train function with parsed arguments
    train(config)


if __name__ == "__main__":
    main()
    # python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml'