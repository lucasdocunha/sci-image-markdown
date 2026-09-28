#!/bin/bash
set -e
cd /home/lucas.ocunha/sci-image-markdown

# Run on GPU 1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME=/media/ssd2/lucas.ocunha/.cache/huggingface
export PYTHONPATH=.

mkdir -p /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/qwen2_5_vl_7b_icdar_loss
mkdir -p logs

echo "Starting Qwen2.5-VL-7B Fine-Tuning with Custom ICDAR Metric Loss on GPU 1..."
exec .venv/bin/python -u train.py --config configs/training/train_7b_icdar_loss.yaml > logs/train_7b_icdar_loss.log 2>&1
