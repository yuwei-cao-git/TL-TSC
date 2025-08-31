#!/bin/bash

LOG_FILE="log_4.txt"
SLEEP_DURATION=10  # seconds

echo "Starting training..." | tee -a "$LOG_FILE"

# List of commands to run
commands=(
	# all head
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'test' --gpus 1 --batch_size 16 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'l' --network 'ResUnet' --lr 1e-3 --optimizer 'adam' --scheduler 'steplr' --loss_func 'mse' --pc_normal True"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc' --log_name 'test_tuned' --gpus 1 --batch_size 16 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'l' --network 'ResUnet' --lr 5e-4 --optimizer 'adam' --scheduler 'onecycle' --loss_func 'mse' --pc_normal True --pretrained_ckpt 'tl_logs/decision_fuse_resunet_pointnetl_rmf_genus_lr1e3_mse_adam_steplr_nokl_normal/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --task 'pc_tsc' --log_name 'test' --gpus 1 --batch_size 16 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'l' --network 'ResUnet' --lr 1e-3 --optimizer 'adam' --scheduler 'steplr' --loss_func 'mse' --pc_normal True"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc' --log_name 'test' --gpus 1 --batch_size 16 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'l' --network 'ResUnet' --lr 1e-3 --optimizer 'adam' --scheduler 'steplr' --loss_func 'mse' --pc_normal True"
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
