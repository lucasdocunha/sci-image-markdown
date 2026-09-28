#!/bin/bash
set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HOME=/media/ssd2/lucas.ocunha/.cache/huggingface
export PYTHONPATH=.

mkdir -p /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/qwen2_5_vl_7b_icdar_loss
mkdir -p /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss
mkdir -p logs

echo "=== [1/2] Fine-Tuning Qwen2.5-VL-7B (Custom ICDAR Loss) ==="
.venv/bin/python -u train.py --config configs/training/train_7b_icdar_loss.yaml 2>&1 | tee logs/train_7b_icdar_loss.log

echo "=== [2/2] Running Evaluation on Test Set (373 samples) ==="
.venv/bin/python -u evaluate.py \
  --config configs/training/train_7b_icdar_loss.yaml \
  --adapter-path /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/qwen2_5_vl_7b_icdar_loss/final_adapter \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss/test_report.json \
  --save-predictions /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss/test_predictions.jsonl 2>&1 | tee logs/eval_7b_icdar_loss.log

echo "=== Pipeline Completed Successfully ==="
