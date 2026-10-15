#!/bin/bash
#SBATCH --job-name=tr-qwen3-8b
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_train_qwen3_8b_%j.out
#SBATCH --error=logs/slurm_train_qwen3_8b_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/qwen3_vl_8b_icdar_loss
mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/qwen3_vl_8b_icdar_loss
mkdir -p logs

echo "=== [1/2] Fine-Tuning Qwen3-VL-8B-Instruct (Custom ICDAR Loss v2) ==="
.venv/bin/python -u train.py --config configs/training/train_qwen3_vl_8b.yaml 2>&1 | tee logs/train_qwen3_vl_8b.log

echo "=== [2/2] Running Evaluation on Test Set (373 samples) ==="
.venv/bin/python -u evaluate.py \
  --config configs/training/train_qwen3_vl_8b.yaml \
  --adapter-path /media/ssd2/datasets/sci-image-markdown-data/outputs/qwen3_vl_8b_icdar_loss/final_adapter \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/qwen3_vl_8b_icdar_loss/test_report.json \
  --save-predictions /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/qwen3_vl_8b_icdar_loss/test_predictions.jsonl 2>&1 | tee logs/eval_qwen3_vl_8b.log

echo "=== Qwen3-VL-8B-Instruct Pipeline Completed Successfully ==="
