#!/bin/bash

LOG_FILE="log_1.txt"
SLEEP_DURATION=10  # seconds

echo "Starting training..." | tee -a "$LOG_FILE"

# List of commands to run
commands=(
    "python train_fuse.py --batch_size 64 --gpu 1 --config "./configs/config_ovf_coarser.yaml" --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --lr 1e-3 --optimizer 'adam' --loss_func "wmse" --dataset "ovf_coarser" --head "fuse_head" --task "tsc" --network 'ResUnet' --scheduler 'steplr' --log_name "tsc_wmse_ovf_coaser_pretraining""

    "python train_fuse.py --batch_size 64 --gpu 1 --config "./configs/config_ovf_coarser.yaml" --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --lr 1e-3 --optimizer 'adam' --loss_func "ewmse" --dataset "ovf_coarser" --head "fuse_head" --task "tsc" --network 'ResUnet' --scheduler 'steplr' --log_name "tsc_ewmse_ovf_coaser_pretraining""
    
    "python finetune.py --cfg 'configs/finetune.yaml' --ft_mode 'freeze_last_1' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_sp/checkpoints/final_model.ckpt'"
    
    "python finetune.py --cfg 'configs/finetune.yaml' --ft_mode 'freeze_last_2' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_sp/checkpoints/final_model.ckpt'"

    "python finetune.py --cfg 'configs/finetune.yaml' --ft_mode 'freeze_last_3' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_sp/checkpoints/final_model.ckpt'"
    
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
