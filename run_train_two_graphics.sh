#!/bin/bash
#SBATCH --job-name=tr-q25vl-two-graphics
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00

set -euo pipefail

# Resilient directory navigation for sicret2 cluster
if [ -d "$HOME/Desktop/sci-image-markdown" ]; then
    cd "$HOME/Desktop/sci-image-markdown"
elif [ -d "$HOME/masters/sci-image-markdown" ]; then
    cd "$HOME/masters/sci-image-markdown"
else
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
fi

source .venv/bin/activate
export HF_HOME=$PWD/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/checkpoints

# Link dataset from /media/ssd2 if generated there on sicret2
if [ -d "/media/ssd2/datasets/sci-image-markdown-data/synthetic_two_graphics" ] && [ ! -d "data/synthetic_two_graphics" ]; then
    mkdir -p data
    ln -sfn /media/ssd2/datasets/sci-image-markdown-data/synthetic_two_graphics data/synthetic_two_graphics
fi

echo "=========================================="
echo "Job Slurm iniciado em: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "GPU alocada: ${CUDA_VISIBLE_DEVICES:-Nenhuma}"
echo "=========================================="

python -u train.py --config configs/training/train_qwen2_5_vl_3b_two_graphics.yaml 2>&1 | tee logs/train_two_graphics.log

echo "=========================================="
echo "Job finalizado com sucesso em: $(date)"
echo "=========================================="
