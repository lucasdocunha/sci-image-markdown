#!/bin/bash
#SBATCH --job-name=tr-q25vl-3b-onfl-eq
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

# On-the-fly equivalence run: 12 epochs x 723 originals with 75% gated degradation
# matches the offline combined run (3 epochs x originals + 3 degraded copies each).
echo "=== treino on-the-fly equivalência (p=0.75 dose offline) iniciado $(date) ==="
python -u train.py --config configs/training/train_qwen2_5_vl_3b_onfly_equiv.yaml 2>&1 | tee logs/train_onfly_equiv.log
echo "=== treino finalizado $(date) ==="
