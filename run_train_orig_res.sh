#!/bin/bash
set -e
cd /home/lucas.ocunha/sci-image-markdown

# Run on GPU 1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME=/media/ssd2/lucas.ocunha/vlm-lab-data/hf-cache
export PYTHONPATH=.

mkdir -p /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs
mkdir -p logs

exec .venv/bin/python -u train.py --config configs/training/train_original_resolution.yaml > logs/train_orig_res.log 2>&1
