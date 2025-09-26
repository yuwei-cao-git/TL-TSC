#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --job-name="ovf-tsc-train"
#SBATCH --time=00:30:00        # Specify run time 

# Trap the exit status of the job
trap 'job_failed=$?' EXIT

# code transfer
cd $SLURM_TMPDIR
mkdir work
cd work
git clone git@github.com:yuwei-cao-git/TL-TSC.git
cd TL-TSC
if [$? -ne 0]; then
  exit 1
fi
echo "Source code cloned!"
# data transfer
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
# tar -I pigz -xf $project/TL-TSC/data/rmf_tl_dataset.tar.gz -C ./data || { echo "rmf extract failed"; exit 1; }
# tar -I pigz -xf $project/TL-TSC/data/ovf_tl_dataset.tar.gz -C ./data || { echo "ovf extract failed"; exit 1; }
# tar -I pigz -xf $project/TL-TSC/data/rmf_superpixel_dataset.tar.gz -C ./data || { echo "rmf extract failed"; exit 1; }
echo "Data transfered"
tar -I pigz -xf $project/TL-TSC/data/ovf_superpixel_dataset.tar.gz -C ./data || { echo "ovf extract failed"; exit 1; }

# Load python module, and additional required modules
echo "load modules"
module load python StdEnv gcc arrow cuda
# module load python StdEnv gcc arrow cuda/12.2 cudnn nccl/2.18.3
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch==2.5.0 pointnext==0.0.5
pip install --no-index timm tensorboardX lightning pytorch_lightning torchaudio==2.5.0 torchdata torcheval torchmetrics torchvision==0.20.0 rasterio imageio wandb pandas
pip install --no-index scikit-learn seaborn open3d==0.18.0 mamba-ssm
echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
wandb login df8a833b419940bc3a6d3e5e04857fe61bb72eef

# Run python script
for dataset in ovf_coarser # ovf_genus
do
	for task in tsca tsc
	do
		for network in ResUnet # Unet # ResNet
		do
			for loss_func in mse wmse
			do
				for sch in steplr onecycle
				do
					for lr in 1e-3 5e-4
					do
						log_name="${task}_${network}_pointnextl_${dataset}_lr${lr}_${loss_func}_${sch}"
						srun python train_fuse.py --gpus 1 --config './configs/config_ovf_coarser.yaml' --data_dir ./data/ovf_superpixel_dataset --dataset "$dataset" --scheduler "$sch" --optimizer "adam" --head "fuse_head" --network "$network" --loss_func "$loss_func" --task "$task" --lr $lr --pc_normal True --log_name "${log_name}"
						cp -r ./tl_logs/${log_name} ~/scratch/tl_logs/
						echo "Logs copied for experiment ${log_name}"
					done
				done
			done
		done
	done
done

echo "theend"
