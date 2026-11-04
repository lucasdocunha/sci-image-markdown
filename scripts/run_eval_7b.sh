#!/bin/bash
set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HOME=/media/ssd2/lucas.ocunha/.cache/huggingface
export PYTHONPATH=.

mkdir -p /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss
mkdir -p logs

echo "Starting Qwen2.5-VL-7B Evaluation on Test Set (373 samples)..."
.venv/bin/python evaluate.py \
  --config configs/training/train_7b_icdar_loss.yaml \
  --adapter-path /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/qwen2_5_vl_7b_icdar_loss/final_adapter \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss/test_report.json \
  --save-predictions /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss/test_predictions.jsonl > logs/eval_7b_icdar_loss.log 2>&1

echo "Evaluation completed! Results saved to /media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/7b_icdar_loss/test_report.json"
