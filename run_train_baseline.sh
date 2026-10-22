#!/bin/bash
#SBATCH --job-name=tr-q25vl-3b
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
cd $HOME/masters/sci-image-markdown
source .venv/bin/activate
export HF_HOME=$PWD/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p outputs/checkpoints

echo "=== treino iniciado $(date) em $(hostname), GPU $CUDA_VISIBLE_DEVICES ==="
python -u train.py --config configs/default.yaml 2>&1 | tee logs/train_baseline.log
echo "=== treino finalizado $(date) ==="
