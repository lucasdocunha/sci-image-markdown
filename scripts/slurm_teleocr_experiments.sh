#!/bin/bash
#SBATCH --job-name=teleocr-expA
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --output=logs/slurm_teleocr_%j.out
#SBATCH --error=logs/slurm_teleocr_%j.err
#SBATCH --time=24:00:00

# Experimento A: TeleOCR zero-shot vs TeleOCR + LoRA (CE loss).
# Stages: [1] zero-shot eval -> [2] LoRA training -> [3] eval of the trained adapter.
# Usage: sbatch scripts/slurm_teleocr_experiments.sh
#        (or skip stages with STAGES="2 3" sbatch ...)

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

STAGES="${STAGES:-1 2 3}"
DATA_ROOT=/media/ssd2/datasets/sci-image-markdown-data
ZS_DIR=$DATA_ROOT/outputs/eval_results/zero_shot_teleocr
TR_DIR=$DATA_ROOT/outputs/teleocr_lora_ce
LORA_EVAL_DIR=$DATA_ROOT/outputs/eval_results/teleocr_lora_ce
mkdir -p "$ZS_DIR" "$TR_DIR" "$LORA_EVAL_DIR" logs

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage 1; then
  echo "=========================================================="
  echo "=== [1/3] TeleOCR Zero-Shot evaluation                 ==="
  echo "=========================================================="
  .venv/bin/python -u evaluate.py \
    --config configs/evaluation/eval_zero_shot_teleocr.yaml \
    --test-file data/processed/test.jsonl \
    --output-report "$ZS_DIR/test_report.json" \
    --save-predictions "$ZS_DIR/test_predictions.jsonl" 2>&1 | tee logs/eval_zero_shot_teleocr.log
fi

if has_stage 2; then
  echo "=========================================================="
  echo "=== [2/3] TeleOCR LoRA fine-tuning (CE loss)           ==="
  echo "=========================================================="
  .venv/bin/python -u train.py --config configs/training/train_teleocr_lora.yaml 2>&1 | tee logs/train_teleocr_lora.log
fi

if has_stage 3; then
  echo "=========================================================="
  echo "=== [3/3] TeleOCR LoRA evaluation on test set          ==="
  echo "=========================================================="
  .venv/bin/python -u evaluate.py \
    --config configs/training/train_teleocr_lora.yaml \
    --adapter-path "$TR_DIR/final_adapter" \
    --test-file data/processed/test.jsonl \
    --output-report "$LORA_EVAL_DIR/test_report.json" \
    --save-predictions "$LORA_EVAL_DIR/test_predictions.jsonl" 2>&1 | tee logs/eval_teleocr_lora.log
fi

echo "=== TeleOCR experiment A completed (stages: $STAGES) ==="
