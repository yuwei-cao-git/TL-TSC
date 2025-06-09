import argparse
from utils.trainer import train
import os
import torch
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def override_config(cfg, args):
    if args.gpus is not None:
        cfg['gpus'] = args.gpus
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.head is not None:
        cfg['head'] = args.head
    if args.dataset is not None:
        cfg['dataset'] = args.dataset
    return cfg

    
def parse_args():
    parser = argparse.ArgumentParser(description="Train model with given parameters")
    # Add arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
    parser.add_argument("--log_name", default="Fuse_resnet_pointnext_rmf")
    parser.add_argument('--gpus', type=int, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--head', type=str, help='Override head option')
    parser.add_argument('--dataset', type=str, help='Override dataset')
    return parser.parse_args()
    
def main():
    # Parse arguments
    args = parse_args()
    cfg = load_config(args.config)
    cfg = override_config(cfg, args)
    cfg['log_name'] = args.log_name
    cfg['save_dir'] = os.path.join(os.getcwd(), 'tl_logs')
    cfg["data_dir"] = (
        args.data_dir
        if args.data_dir is not None
        else os.path.join(os.getcwd(), "data")
    )
    
    class_weights = cfg.get('class_weights', None)
    cfg['class_weights'] = torch.tensor(class_weights).float()
    
    os.makedirs(cfg['save_dir'], exist_ok=True)
    print(cfg)
    # Call the train function with parsed arguments
    train(cfg)


if __name__ == "__main__":
    main()
    # python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml'