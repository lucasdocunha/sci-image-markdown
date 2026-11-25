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

source .venv/bin/activate
export HF_HOME="/media/ssd2/lucas.ocunha/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

STAGES="${STAGES:-1 2 3}"
DATA_ROOT=/media/ssd2/datasets/sci-image-markdown-data
ZS_DIR=$DATA_ROOT/outputs/eval_results/zero_shot_teleocr
TR_DIR=$DATA_ROOT/outputs/teleocr_lora_ce
LORA_EVAL_DIR=$DATA_ROOT/outputs/eval_results/teleocr_lora_ce
META="data/raw/test_metadata.jsonl"
mkdir -p "$ZS_DIR" "$TR_DIR" "$LORA_EVAL_DIR" logs

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage 1; then
  echo "=========================================================="
  echo "=== [1/3] TeleOCR Zero-Shot evaluation                 ==="
  echo "=========================================================="
  python -u evaluate.py \
    --config configs/evaluation/eval_zero_shot_teleocr.yaml \
    --test-file data/processed/test.jsonl \
    --output-report "$ZS_DIR/test_report.json" \
    --save-predictions "$ZS_DIR/test_predictions.jsonl" 2>&1 | tee logs/eval_zero_shot_teleocr.log

  if [ -f "$META" ] && [ -f "$ZS_DIR/test_predictions.jsonl" ]; then
    echo "=== Running official metric analysis for Zero-Shot TeleOCR ==="
    python analyze_errors.py \
      --metadata "$META" \
      --repo-predictions "$ZS_DIR/test_predictions.jsonl" \
      --out-dir "$ZS_DIR" 2>&1 | tee -a logs/eval_zero_shot_teleocr.log
  fi
fi

if has_stage 2; then
  echo "=========================================================="
  echo "=== [2/3] TeleOCR LoRA fine-tuning (CE loss)           ==="
  echo "=========================================================="
  python -u train.py --config configs/training/train_teleocr_lora.yaml 2>&1 | tee logs/train_teleocr_lora.log
fi

if has_stage 3; then
  echo "=========================================================="
  echo "=== [3/3] TeleOCR LoRA evaluation on test set          ==="
  echo "=========================================================="
  python -u evaluate.py \
    --config configs/training/train_teleocr_lora.yaml \
    --adapter-path "$TR_DIR/final_adapter" \
    --test-file data/processed/test.jsonl \
    --output-report "$LORA_EVAL_DIR/test_report.json" \
    --save-predictions "$LORA_EVAL_DIR/test_predictions.jsonl" 2>&1 | tee logs/eval_teleocr_lora.log

  if [ -f "$META" ] && [ -f "$LORA_EVAL_DIR/test_predictions.jsonl" ]; then
    echo "=== Running official metric analysis for LoRA TeleOCR ==="
    python analyze_errors.py \
      --metadata "$META" \
      --repo-predictions "$LORA_EVAL_DIR/test_predictions.jsonl" \
      --out-dir "$LORA_EVAL_DIR" 2>&1 | tee -a logs/eval_teleocr_lora.log
  fi
fi

echo "=== TeleOCR experiment A completed (stages: $STAGES) ==="

echo "=== Releasing held train_fourier jobs in Slurm queue ==="
scontrol release 447,448,449,453,454,455,456,457,458,459,461,462,463,464 || true
squeue
