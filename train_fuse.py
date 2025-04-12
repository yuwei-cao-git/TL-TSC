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
    parser.add_argument("--batch_size", type=int, default=8, help="Number of epochs to train the model")
    parser.add_argument("--tile_size", default=64, type=int, choices=[32, 64, 128])
    parser.add_argument("--log_name", default="Fuse_ff_mamba_pointnext_b_Unet_10")
    
def main():
    # Parse arguments
    args = parse_args()
    config = get_config(args.config)
    config['log_name'] = args.log_name
    config['save_dir'] = os.path.join(os.getcwd(), 'tl_logs')
    config["data_dir"] = (
        args["data_dir"]
        if args["data_dir"] is not None
        else os.path.join(os.getcwd(), "data", f"{config['dataset']}_tl_dataset")
    )
    config['dataset'] = config['mode'] if config['mode'] in ['rmf', 'ovf'] else 'rmf+ovf'
    if config['weighted_loss']:
        config['class_weights_img'] = torch.from_numpy(np.array(config['class_weights_img'])).float()
        config['class_weights_pc'] = torch.from_numpy(np.array(config['class_weights_pc'])).float()

    os.makedirs(config['save_dir'], exist_ok=True)
    print(config)

    # Call the train function with parsed arguments
    train(config)


if __name__ == "__main__":
    main()
    # python train_fuse.py --dataset 'rmf' --data_dir '/mnt/g/rmf/rmf_tl_dataset' --pc_transforms True --spatial_attention --tile_size 64 --linear_layers_dims 256,128 
    # python train_fuse.py --dataset 'ovf' --data_dir '/mnt/g/ovf/ovf_tl_dataset' --pc_transforms True --spatial_attention --tile_size 64 --linear_layers_dims 256,128 