#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=LSDF
#SBATCH --gres=gpu:full:1
#SBATCH --mem="192000"

# Clear modules and load CUDA 12.9
module purge
module load devel/cuda/12.9

# Link environment variables
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

config=$1

# Activate Python 3.11 environment
source ~/envs/mamba_env/bin/activate

# Execute script
python src/Train.py "$config"