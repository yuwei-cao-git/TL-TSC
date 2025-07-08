#!/bin/bash

LOG_FILE="log_2.txt"
SLEEP_DURATION=10  # seconds

echo "Starting training..." | tee -a "$LOG_FILE"

# List of commands to run
commands=(
    # no-pc head
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc' --log_name 'no_pc_head_unet_pointnetb_ovf_genus_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 16 --head 'no_pc_head' --dataset 'ovf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_genus.yaml' --task 'tsc' --log_name 'no_pc_head_unet_pointnetb_rmf_genus_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 16 --head 'no_pc_head' --dataset 'rmf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc' --log_name 'no_pc_head_unet_pointnetb_ovf_genus_normal_lr1e4_wmse_cosine_tuned' --gpus 1 --batch_size 16 --head 'no_pc_head' --dataset 'ovf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_pc_head_unet_pointnetb_rmf_genus_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"
    
    # all head
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc' --log_name 'all_head_unet_pointnetb_ovf_genus_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 16 --head 'all_head' --dataset 'ovf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_genus.yaml' --task 'tsc' --log_name 'all_head_unet_pointnetb_rmf_genus_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 16 --head 'all_head' --dataset 'rmf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc' --log_name 'all_head_unet_pointnetb_ovf_genus_normal_lr1e4_wmse_cosine_tuned' --gpus 1 --batch_size 16 --head 'all_head' --dataset 'ovf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/all_head_unet_pointnetb_rmf_genus_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"
    
    # no-img head
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc' --log_name 'no_img_head_unet_pointnetb_ovf_genus_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 16 --head 'no_img_head' --dataset 'ovf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_genus.yaml' --task 'tsc' --log_name 'no_img_head_unet_pointnetb_rmf_genus_normal_lr1e4_wmse_cosine' --gpus 1 --batch_size 16 --head 'no_img_head' --dataset 'rmf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse'"
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc' --log_name 'no_img_head_unet_pointnetb_ovf_genus_normal_lr1e4_wmse_cosine_tuned' --gpus 1 --batch_size 16 --head 'no_img_head' --dataset 'ovf_genus' --network 'Unet' --lr 1e-4 --loss_func 'wmse' --pretrained_ckpt 'tl_logs/no_img_head_unet_pointnetb_rmf_genus_normal_lr1e4_wmse_cosine/checkpoints/final_model.ckpt'"
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