import argparse
import os
import torch
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def override_config(cfg, args):
    if args.task is not None:
        cfg['task'] = args.task
    if args.gpus is not None:
        cfg['gpus'] = args.gpus
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.head is not None:
        cfg['head'] = args.head
    if args.encoder is not None:
        cfg['encoder'] = args.encoder
    if args.network is not None:
        cfg['network'] = args.network
    if args.dataset is not None:
        cfg['dataset'] = args.dataset
    if args.test_dataset is not None:
        cfg["test_dataset"] = args.test_dataset
    if args.pretrained_ckpt is not None:
        cfg['pretrained_ckpt'] = args.pretrained_ckpt
    if args.align_header is not None:
        cfg['align_header'] = args.align_header
    if args.scheduler is not None:
        cfg['scheduler'] = args.scheduler
    # hps
    if args.pc_lr is not None:
        cfg["pc_lr"] = args.pc_lr
    if args.img_lr is not None:
        cfg["img_lr"] = args.img_lr
    if args.fuse_lr is not None:
        cfg["fuse_lr"] = args.fuse_lr
    if args.emb_dims is not None:
        cfg['emb_dims'] = args.emb_dims
    if args.optimizer is not None:
        cfg['optimizer'] = args.optimizer
    if args.multitasks_uncertain_loss:
        cfg['multitasks_uncertain_loss'] = args.multitasks_uncertain_loss
    if args.loss_func is not None:
        cfg['loss_func'] = args.loss_func
    if args.pc_normal:
        cfg['pc_normal'] = args.pc_normal
    if args.use_ms:
        cfg['use_ms'] = args.use_ms
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with given parameters")
    # Add arguments
    parser.add_argument('--task', type=str)
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
    parser.add_argument("--test_data_dir", type=str, default=None, help="path to test data dir")
    parser.add_argument("--log_name", default="Fuse_resnet_pointnext_rmf")
    parser.add_argument('--gpus', type=int, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--head', type=str, help='Override head option')
    parser.add_argument('--network', type=str, help='Override head option')
    parser.add_argument('--encoder', type=str, help='Override encoder option')
    parser.add_argument('--dataset', type=str, help='Override dataset')
    parser.add_argument("--test_dataset", type=str, help="Override test dataset")
    parser.add_argument('--pc_lr', type=float)
    parser.add_argument("--img_lr", type=float)
    parser.add_argument("--fuse_lr", type=float)
    parser.add_argument('--multitasks_uncertain_loss', type=bool, default=False)
    parser.add_argument('--loss_func', type=str)
    parser.add_argument('--emb_dims', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--pretrained_ckpt', default=None)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--pc_normal', type=bool, default=False)
    parser.add_argument('--use_ms', type=bool, default=False) 
    parser.add_argument('--level', type=str)
    parser.add_argument('--align_header', type=str)
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
    cfg["test_data_dir"] = (
        args.test_data_dir
        if args.test_data_dir is not None
        else os.path.join(os.getcwd(), "data")
    )
    if cfg["loss_func"] in ["wmse", "wrmse", "wkl", "ewmse"]:
        class_weights = cfg.get(f'{args.dataset}_class_weights', None)
        cfg[f'{args.dataset}_class_weights'] = torch.tensor(class_weights).float()
    else:
        cfg[f'{args.dataset}_class_weights'] = None

    os.makedirs(cfg['save_dir'], exist_ok=True)
    print(cfg)
    # Call the train function with parsed arguments
    if cfg['task'] == 'tsc_cd':
        from utils.cd_trainer import train
        train(cfg, args.level)
    else:
        from utils.trainer import train
        train(cfg)


if __name__ == "__main__":
    main()
    # python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml'
