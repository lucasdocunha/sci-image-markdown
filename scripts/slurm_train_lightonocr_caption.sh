#!/bin/bash
#SBATCH --job-name=lighton-cap
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --output=logs/slurm_lighton_caption_%j.out
#SBATCH --error=logs/slurm_lighton_caption_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

source .venv/bin/activate
export HF_HOME="/media/ssd2/lucas.ocunha/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

DATA_ROOT=/media/ssd2/datasets/sci-image-markdown-data
OUT_DIR=$DATA_ROOT/outputs/lightonocr_3_4b_caption_icdar
EVAL_DIR=$DATA_ROOT/outputs/eval_results/lightonocr_3_4b_caption_icdar
META="data/raw/test_metadata.jsonl"
mkdir -p "$OUT_DIR" "$EVAL_DIR" logs

echo "=========================================================="
echo "=== [1/2] Fine-Tuning LightOnOCR-3-4B + Caption (ICDAR Loss v3) ==="
echo "=========================================================="
python -u train.py --config configs/training/train_lightonocr_3_4b_caption.yaml 2>&1 | tee logs/train_lightonocr_3_4b_caption.log

echo "=========================================================="
echo "=== [2/2] Running Evaluation on Test Set (with Captions) ==="
echo "=========================================================="
python -u evaluate.py \
  --config configs/training/train_lightonocr_3_4b_caption.yaml \
  --adapter-path "$OUT_DIR/final_adapter" \
  --test-file data/processed/test_with_caption.jsonl \
  --output-report "$EVAL_DIR/test_report.json" \
  --save-predictions "$EVAL_DIR/test_predictions.jsonl" 2>&1 | tee logs/eval_lightonocr_3_4b_caption.log

if [ -f "$META" ] && [ -f "$EVAL_DIR/test_predictions.jsonl" ]; then
  echo "=== Running official metric analysis (605 panels) ==="
  python analyze_errors.py \
    --metadata "$META" \
    --repo-predictions "$EVAL_DIR/test_predictions.jsonl" \
    --out-dir "$EVAL_DIR" 2>&1 | tee -a logs/eval_lightonocr_3_4b_caption.log
fi

echo "=== Pipeline Completed Successfully ==="
