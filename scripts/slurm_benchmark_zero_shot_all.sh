#!/bin/bash
#SBATCH --job-name=vlm-zshot
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_zero_shot_%j.out
#SBATCH --error=logs/slurm_zero_shot_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_lightonocr_3_4b
mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_qwen3_vl_4b
mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_qwen3_vl_8b
mkdir -p logs

echo "=========================================================="
echo "=== [1/3] Benchmarking LightOnOCR-3-4B (Zero-Shot)     ==="
echo "=========================================================="
.venv/bin/python -u evaluate.py \
  --config configs/evaluation/eval_zero_shot_lightonocr.yaml \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_lightonocr_3_4b/test_report.json \
  --save-predictions /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_lightonocr_3_4b/test_predictions.jsonl 2>&1 | tee logs/eval_zero_shot_lightonocr.log

echo "=========================================================="
echo "=== [2/3] Benchmarking Qwen3-VL-4B-Instruct (Zero-Shot) ==="
echo "=========================================================="
.venv/bin/python -u evaluate.py \
  --config configs/evaluation/eval_zero_shot_qwen3_4b.yaml \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_qwen3_vl_4b/test_report.json \
  --save-predictions /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_qwen3_vl_4b/test_predictions.jsonl 2>&1 | tee logs/eval_zero_shot_qwen3_4b.log

echo "=========================================================="
echo "=== [3/3] Benchmarking Qwen3-VL-8B-Instruct (Zero-Shot) ==="
echo "=========================================================="
.venv/bin/python -u evaluate.py \
  --config configs/evaluation/eval_zero_shot_qwen3_8b.yaml \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_qwen3_vl_8b/test_report.json \
  --save-predictions /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/zero_shot_qwen3_vl_8b/test_predictions.jsonl 2>&1 | tee logs/eval_zero_shot_qwen3_8b.log

echo "=== All Zero-Shot Evaluations Completed Successfully ==="
