#!/bin/bash
set -euo pipefail
cd /home/lucas/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

/home/lucas/miniconda3/envs/DeepLearning/bin/python evaluate.py \
  --config configs/default.yaml \
  --training-config configs/evaluation/original_resolution.yaml \
  --adapter-path outputs/checkpoints_icdar_loss/final_adapter \
  --test-file data/processed/test.jsonl \
  --output-report outputs/eval_results/icdar_loss_orig_res_test_report.json \
  --save-predictions outputs/eval_results/icdar_loss_orig_res_test_predictions.jsonl
