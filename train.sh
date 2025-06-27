#!/bin/bash

python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_combined.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_csp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_csp' --network 'Unet' --lr 1e-4 --loss_func 'wrmse'

python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_combined.yaml' --log_name 'all_head_unet_pointnetb_ovf_csp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_csp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'

python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_combined.yaml' --log_name 'no_img_head_unet_pointnetb_rmf_csp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'rmf_csp' --network 'Unet' --lr 1e-4 --loss_func 'wrmse'

python train_fuse.py --data_dir '/mnt/g/csp' --config 'configs/config_combined.yaml' --log_name 'no_img_head_unet_pointnetb_csp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'csp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'

python train_fuse.py --data_dir '/mnt/g/csp' --config 'configs/config_combined.yaml' --log_name 'all_head_unet_pointnetb_csp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'csp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'

#python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_sp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_superpixel_dataset' --network 'Unet' --lr 1e-4 --loss_func 'wmse'

#python train_fuse.py --data_dir '/mnt/g/rmf' --config 'configs/config_rmf.yaml' --log_name 'no_img_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'rmf_superpixel_dataset' --network 'Unet' --lr 1e-4 --loss_func 'wmse'

#python train_fuse.py --data_dir '/mnt/g/ovf' --config 'configs/config_ovf.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_sp_lr1e4_wmse_cosine_pretrained' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_superpixel_dataset' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_img_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'

python train_fuse.py --data_dir '/mnt/g/rmf' --config 'configs/config_rmf.yaml' --log_name 'no_img_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine_pretrained_ovf_csp' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'rmf_superpixel_dataset' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_img_head_unet_pointnetb_ovf_csp_lr1e4_wmse_cosine_bnlayer/checkpoints/final_model.ckpt'

python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'all_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'