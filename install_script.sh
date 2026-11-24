#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=LSDF
#SBATCH --gres=gpu:full:1
#SBATCH --mem="16000"

# 1. Clear modules and load CUDA 12.9
module purge
module load devel/cuda/12.9

# 2. Configure system paths
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. Force GNU compilers
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)

# 4. BYPASS CUDA VERSION MISMATCH ASSERTION IN PYTORCH
export FORCE_CUDA="1"
export ALLOW_CUDA_MISMATCH=1
export TORCH_CUDA_TOO_NEW_FAIL_WITH_ERROR=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# 5. Create environment if it doesn't exist and activate it
ENV_PATH="$HOME/envs/mamba_env"
if [ ! -d "$ENV_PATH" ]; then
    python3.11 -m venv "$ENV_PATH"
fi
source "$ENV_PATH/bin/activate"

# 6. Install all build dependencies and runtime dependencies
pip install --upgrade pip setuptools wheel packaging==26.2 ninja==1.13.0 einops==0.8.2 numpy==2.4.6 huggingface_hub==1.25.1 triton==3.7.1 transformers==5.14.1

# 7. Install newest stable PyTorch built for CUDA 12
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-cache-dir

# 8. Build causal-conv1d and mamba-ssm
pip install causal-conv1d==1.6.2.post1 --no-build-isolation --no-deps --no-cache-dir --force-reinstall -v
pip install mamba-ssm==2.3.2.post1 --no-build-isolation --no-deps --no-cache-dir --force-reinstall -v

# 9. Install remaining packages from requirements.txt
pip install -r requirements.txt