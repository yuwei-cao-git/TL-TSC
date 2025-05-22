import argparse
from utils.trainer import train
import os
import torch
import yaml

def get_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train model with given parameters")
    # Add arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
    parser.add_argument("--log_name", default="Fuse_resnet_pointnext_rmf")
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
    
    class_weights = config.get('class_weights', None)
    config['class_weights'] = torch.tensor(class_weights).float()
    
    os.makedirs(config['save_dir'], exist_ok=True)
    print(config)

    # Call the train function with parsed arguments
    train(config)


if __name__ == "__main__":
    main()
    # python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml'