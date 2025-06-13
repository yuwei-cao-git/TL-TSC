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
    if args.network is not None:
        cfg['network'] = args.network
    if args.dataset is not None:
        cfg['dataset'] = args.dataset
    # hps
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.multitasks_uncertain_loss is not None:
        cfg['multitasks_uncertain_loss'] = args.multitasks_uncertain_loss
    if args.weighted_cls_loss is not None:
        cfg['weighted_cls_loss'] = args.weighted_cls_loss
    if args.loss_func is not None:
        cfg['loss_func'] = args.loss_func
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
    parser.add_argument('--network', type=str, help='Override head option')
    parser.add_argument('--dataset', type=str, help='Override dataset')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--multitasks_uncertain_loss', type=bool, default=True)
    parser.add_argument('--weighted_cls_loss', type=bool, default=False)
    parser.add_argument('--loss_func', type=str)
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
    if cfg['weighted_cls_loss']:
        class_weights = cfg.get('class_weights', None)
        cfg['class_weights'] = torch.tensor(class_weights).float()
    else:
        cfg['class_weights'] = None
    
    os.makedirs(cfg['save_dir'], exist_ok=True)
    print(cfg)
    # Call the train function with parsed arguments
    train(cfg)


if __name__ == "__main__":
    main()
    # python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml'