#!/bin/bash
#SBATCH --job-name=tr-q25vl-3b-aug
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=/home/adryan.silva/masters/sci-image-markdown/logs/train_%j.out
#SBATCH --error=/home/adryan.silva/masters/sci-image-markdown/logs/train_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
cd $HOME/masters/sci-image-markdown
source .venv/bin/activate
export HF_HOME=$PWD/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/checkpoints

echo "=== treino combinado (originais + cópias degradadas) iniciado $(date) ==="
python -u scripts/build_train_combined.py --overwrite
echo "=== $(wc -l < data/augmented/train_combined.jsonl) registros combinados ==="
python -u train.py --config configs/training/train_qwen2_5_vl_3b_aug.yaml 2>&1 | tee logs/train_augmented.log
echo "=== treino finalizado $(date) ==="
