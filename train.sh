#!/bin/bash

LOG_FILE="log_5.txt"
SLEEP_DURATION=10  # seconds

echo "Starting training..." | tee -a "$LOG_FILE"

# List of commands to run
commands=(
	# all head
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --task 'tsc_aligned' --log_name 'tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --dp 0.3"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'ovf_coarser_tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --dp 0.3 --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --task 'tsc_aligned' --log_name 'tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_wmse' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'wmse' --scheduler 'steplr' --dp 0.3"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --task 'tsc_aligned' --log_name 'tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_ewmse' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'ewmse' --scheduler 'steplr' --dp 0.3"
    
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_mid_decision' --log_name 'test' --gpus 1 --batch_size 32 --head 'all_head' --dataset 'ovf_coarser' --encoder 'b' --network 'FCNResNet' --lr 5e-4 --emb_dims 768 --loss_func 'wmse' --scheduler 'steplr' --multitasks_uncertain_loss True --dp 0.5"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf.yaml' --task 'tsc_mid_decision' --log_name 'tsc_mf_df_rmf_sp_uncertainloss_retnetfcn_pointnextb_wmse' --gpus 1 --batch_size 32 --head 'all_head' --dataset 'rmf_sp' --encoder 'b' --network 'FCNResNet' --lr 5e-4 --emb_dims 768 --loss_func 'wmse' --scheduler 'steplr' --multitasks_uncertain_loss True"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_mid_decision' --log_name 'test' --gpus 1 --batch_size 32 --head 'all_head' --dataset 'ovf_coarser' --encoder 'b' --network 'FCNResNet' --lr 5e-4 --emb_dims 768 --loss_func 'wmse' --scheduler 'steplr' --multitasks_uncertain_loss True --dp 0.5 --pretrained_ckpt 'tl_logs/tsc_mf_df_rmf_sp_uncertainloss_retnetfcn_pointnextb_wmse/checkpoints/final_model.ckpt'"
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
