#!/bin/bash
#SBATCH --job-name=oasis-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

set -euo pipefail

cd "$HOME/comp3710_lab2"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ./pytorch-env

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export UNET_DATA_ROOT="${UNET_DATA_ROOT:-/home/groups/comp3710/OASIS}"
export UNET_IMG="${UNET_IMG:-128}"
export UNET_EPOCHS="${UNET_EPOCHS:-30}"
export UNET_BATCH="${UNET_BATCH:-8}"
export UNET_STAGE="${UNET_STAGE:-full}"
export UNET_OUTPUT="${UNET_OUTPUT:-unet_outputs}"
export UNET_WORKERS="${UNET_WORKERS:-2}"

mkdir -p logs "$UNET_OUTPUT"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

jupyter nbconvert --to notebook --execute --inplace part4_unet.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
