#!/bin/bash

<<<<<<< HEAD
LOG_FILE="log.txt"
=======
LOG_FILE="log_7.txt"
>>>>>>> 5ae178ce39bfbf34d5c3d07e16c01653142d71fe
SLEEP_DURATION=10  # seconds

echo "Starting training..." | tee -a "$LOG_FILE"

# List of commands to run
commands=(
<<<<<<< HEAD
    # no_nomal
    # pre-training: rmf_sp
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_sp
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_sp' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_genus
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_genus' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_genus' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_coarser
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_coarser' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_4class
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_4class' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # tuning using pre-trained tsc_aligned_rmf_sp
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_sp_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_genus_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_genus' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_coarser_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"

    # for comparison with ft-full inint
    "python finetune.py --cfg 'configs/finetune.yaml' --ft_mode 'full_ft' --pretrained_ckpt 'tsc_aligned_ovf_genus/checkpoints/final_model.ckpt'" 

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_4class_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"

    # tuning using pre-trained tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_normal (+ normal)
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_sp_normal_tuned' --pc_normal True --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_normal/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_coarser_normal_tuned' --pc_normal True --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_normal/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_pretained_ovf_4class_tuned' --pc_normal True --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_normal/checkpoints/final_model.ckpt'"

    # pre-training: rmf_genus
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_genus' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_genus' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # tuning using pre-trained tsc_aligned_rmf_genus
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_genus_pretained_ovf_sp_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_genus/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_genus_pretained_ovf_genus_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_genus' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_genus/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_genus_pretained_ovf_coarser_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_genus/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_genus_pretained_ovf_4class_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_genus/checkpoints/final_model.ckpt'"

    # pre-training: rmf_4class
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_4class' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # tuning using pre-trained tsc_aligned_rmf_4class
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_4class_pretained_ovf_sp_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_4class/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_4class_pretained_ovf_genus_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_genus' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_4class/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_4class_pretained_ovf_coarser_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_4class/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_4class_pretained_ovf_4class_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_4class/checkpoints/final_model.ckpt'"

    # tuning rmf_sp using pretained ovf_sp
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_sp_pretained_rmf_sp_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"

    # tuning rmf_genus using pretained ovf_sp
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_sp_pretained_rmf_genus_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_genus' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"  

    # tuning rmf_4class using pretained ovf_sp
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_sp_pretained_rmf_4class_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tsc_aligned_ovf_sp/checkpoints/final_model.ckpt'"  

    # tuning rmf_genus using pretained ovf_genus
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_genus_pretained_rmf_genus_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_genus' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tsc_aligned_ovf_genus/checkpoints/final_model.ckpt'"  

    # with normal
    # training from scratch: ovf_sp
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_sp_normal' --gpus 1 --pc_normal True --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_genus
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_genus.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_genus_normal' --pc_normal True --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_genus' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_coarser
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_coarser_normal' --pc_normal True --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"

    # training from scratch: ovf_4class
    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --pc_normal True --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_ovf_4class_normal' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'" 

    # pre-training rmf_sp with ewmse
    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_ewmse' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'ewmse' --scheduler 'steplr'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_ewmse_ovf_coarser_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_sp_ewmse/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_4class.yaml' --task 'tsc_aligned' --log_name 'tsc_aligned_rmf_sp_ewmse_ovf_4class_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_4class' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_aligned_rmf_sp_ewmse/checkpoints/final_model.ckpt'"
=======
    "python finetune.py --cfg 'configs/finetune.yaml' --ft_mode 'adapters'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_sp.yaml' --task 'tsc_aligned' --log_name 'ovf_sp_tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_normal/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'test' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr'"
    
    "python finetune.py --cfg 'configs/finetune.yaml' --ft_mode 'full_ft'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'test' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 5e-4 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pc_normal True"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --pc_normal True --log_name 'ovf_coarser_normal_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_mse_weighted_normal/checkpoints/final_model.ckpt'"

    "python train_fuse.py --data_dir '/mnt/g/rmf/rmf_superpixel_dataset' --config 'configs/config_rmf_sp.yaml' --task 'tsc_aligned' --log_name 'tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_ewmse_weighted' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'rmf_sp' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'ewmse' --scheduler 'steplr'"

    "python train_fuse.py --data_dir '/mnt/g/ovf/ovf_superpixel_dataset' --config 'configs/config_ovf_coarser.yaml' --task 'tsc_aligned' --log_name 'ovf_coarser_tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_ewmse_tuned' --gpus 1 --batch_size 32 --head 'fuse_head' --dataset 'ovf_coarser' --encoder 'b' --network 'ResUnet' --lr 1e-3 --emb_dims 768 --loss_func 'mse' --scheduler 'steplr' --pretrained_ckpt 'tl_logs/tsc_ResUnet_aligned_noms_pointnext_b_rmf_sp_ewmse_weighted/checkpoints/final_model.ckpt'"
>>>>>>> 5ae178ce39bfbf34d5c3d07e16c01653142d71fe
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
