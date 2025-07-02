#!/bin/bash

LOG_FILE="log.txt"
SLEEP_DURATION=10  # seconds

echo "Starting training..." | tee -a "$LOG_FILE"

# List of commands to run
commands=(
    # rmf_sp
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'all_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pc_normal False"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'no_img_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'no_img_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pc_normal False"

    # ovf_sp
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pc_normal False"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_sp_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pc_normal False"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    # tuning ovf_sp on pretrained rmf_sp
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_lr1e4_wmse_cosine_pretrained_rmf_sp' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/all_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine/checkpoints/final_model.ckpt' --pc_normal False"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_sp_lr1e4_wmse_cosine_pretrained_rmf_sp' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_img_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine/checkpoints/final_model.ckpt' --pc_normal False"

    # tuning ovf_sp_normal on pretrained rmf_sp_normal
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'no_img_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine_pretrained_rmf_sp_normal' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_img_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine_pretrained_rmf_sp_normal' --gpus 1 --batch_size 8 --head 'no_img_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/all_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"

    # tune rmf_sp on ovf_csp
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'all_head_unet_pointnetb_rmf_sp_lr1e4_wmse_cosine_pretrained_ovf_csp' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/all_head_unet_pointnetb_ovf_csp_lr1e4_wmse_cosine/checkpoints/final_model.ckpt' --pc_normal False"

    # fuse_head
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'fuse_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'fuse_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'fuse_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'fuse_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'fuse_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine_pretrained_rmf_sp_normal' --gpus 1 --batch_size 8 --head 'fuse_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/fuse_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"

    # no_pc_head
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --log_name 'no_pc_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_pc_head' --dataset 'rmf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'no_pc_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'no_pc_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'no_pc_head_unet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine_pretrained_rmf_sp_normal' --gpus 1 --batch_size 8 --head 'no_pc_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_pc_head_unet_pointnetb_rmf_sp_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_resnet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'ResNet' --lr 1e-4 --loss_func 'wmse'"

    # tuning ovf_sp_normal
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_resnet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'ResNet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_ResUnet_pointnetb_ovf_sp_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'ResUnet' --lr 1e-4 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_normal_lr1e5_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-5 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_normal_lr5e3_wmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'ResNet' --lr 5e-3 --loss_func 'wmse'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf.yaml' --log_name 'all_head_unet_pointnetb_ovf_sp_normal_lr1e4_wrmse_cosine' --gpus 1 --batch_size 8 --head 'all_head' --dataset 'ovf_sp' --network 'Unet' --lr 1e-4 --loss_func 'wrmse'"
)

# Loop over and run each command
for cmd in "${commands[@]}"; do
    echo -e "\n\n[START $(date)] Running: $cmd" | tee -a "$LOG_FILE"
    eval $cmd >> "$LOG_FILE" 2>&1
    echo -e "[END $(date)] Finished: $cmd\n" | tee -a "$LOG_FILE"
    echo "Sleeping for $SLEEP_DURATION seconds..." | tee -a "$LOG_FILE"
    sleep $SLEEP_DURATION
done

echo -e "\nAll training runs completed. ($(date))" | tee -a "$LOG_FILE"