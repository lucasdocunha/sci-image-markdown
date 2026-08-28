#!/bin/bash
set -euo pipefail
cd /home/lucas/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "==> Exportando e fundindo pesos LoRA fine-tuned com o modelo base..."
exec /home/lucas/miniconda3/envs/DeepLearning/bin/python export.py \
    --config configs/default.yaml \
    --adapter-path outputs/checkpoints/final_adapter \
    --output-dir outputs/merged_model
