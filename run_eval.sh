#!/bin/bash
#SBATCH --job-name=ev-q25vl
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=06:00:00

# usage: sbatch run_eval.sh <TAG> <ADAPTER_PATH> <TEST_JSONL> [TRAINING_CONFIG]
set -euo pipefail
TAG=$1
ADAPTER=$2
TEST=$3
TRAINING_CONFIG=${4:-}

cd $HOME/masters/sci-image-markdown
source .venv/bin/activate
export HF_HOME=$PWD/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
OUT="outputs/eval_results/$TAG"
mkdir -p "$OUT"

echo "=== eval [$TAG] iniciado: adapter=$ADAPTER, test=$TEST, $(date) ==="
if [ -n "$TRAINING_CONFIG" ]; then
    python -u evaluate.py \
        --config configs/default.yaml \
        --training-config "$TRAINING_CONFIG" \
        --adapter-path "$ADAPTER" \
        --test-file "$TEST" \
        --output-report "$OUT/report.json" \
        --save-predictions "$OUT/predictions.jsonl" 2>&1 | tee "logs/eval_$TAG.log"
else
    python -u evaluate.py \
        --config configs/default.yaml \
        --adapter-path "$ADAPTER" \
        --test-file "$TEST" \
        --output-report "$OUT/report.json" \
        --save-predictions "$OUT/predictions.jsonl" 2>&1 | tee "logs/eval_$TAG.log"
fi
echo "=== eval [$TAG] finalizado: $OUT/report.json, $(date) ==="
