#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --job-name="multi-gpu-tl-train"
#SBATCH --time=10:00:00        # Specify run time 

# Trap the exit status of the job
trap 'job_failed=$?' EXIT

# code transfer
cd $SLURM_TMPDIR
mkdir work
cd work
git clone git@github.com:yuwei-cao-git/TL-TSC.git
cd TL-TSC
echo "Source code cloned!"

# data transfer
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/TL-TSC/data/ovf_tl_dataset.tar -C ./data
tar -I pigz -xf $project/TL-TSC/data/ovf_superpixel_dataset.tar.gz -C ./data || { echo "ovf extract failed"; exit 1; }
ls ./data
echo "Data transfered"

# Load python module, and additional required modules
echo "load modules"
#module load python StdEnv gcc arrow
module load python StdEnv gcc arrow cuda
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index lightning pytorch_lightning torch torchaudio torchdata torcheval torchmetrics torchvision
pip install pointnext==0.0.5 
pip install --no-index ray[tune] 
pip install --no-index mamba-ssm tensorboardX scikit-learn seaborn rasterio imageio wandb

echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
wandb login *

# Run python script
# Define your resolution list
# Run your model multiple timesi
log_name = "mambafuse_pointnext_resunet_s4_9bands_compose"
srun python train_img.py \
	--data_dir './data' \
	--batch_size 64 \
	--img_transforms "compose" \
	--remove_bands \
	--weighted_loss False \
	--mamba_fuse \
	--log_name "${log_name}"
	
# Package logs: adjust paths if logs are saved elsewhere
# Create output directory
mkdir -p ~/scratch/tl_logs/${log_name}
echo "Created output dir: ~/scratch/tl_logs/${log_name}"
if [ -d "./tl_logs/${log_name}" ]; then
	tar -cf ~/scratch/img_logs/${log_name}/logs.tar ./tl_logs/${log_name}
	echo "Logs archived for experiment ${log_name}"
else
	echo "Warning: Log directory ./tl_logs/${log_name} does not exist, skipping tar."

echo "theend"
