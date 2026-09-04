#!/bin/bash
set -euo pipefail
cd /home/lucas/masters/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "==> Exportando e fundindo pesos LoRA fine-tuned com o modelo base..."
exec .venv/bin/python export.py \
    --config configs/default.yaml \
    --adapter-path outputs/checkpoints/final_adapter \
    --output-dir outputs/merged_model
