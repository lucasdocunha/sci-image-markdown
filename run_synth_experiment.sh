#!/bin/bash
# Compare training on real data only vs. real data + ~15% synthetic two-table samples.
# Same seed, hyperparameters, and test set (data/processed/test.jsonl) for both runs.
set -euo pipefail

cd /home/lucas/masters/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PYTHON=".venv/bin/python"
TEST_FILE="data/processed/test.jsonl"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "[1/5] Building synthetic pairs (side_by_side + stacked)"
$PYTHON make_synthetic_pairs.py --layout side_by_side --num-pairs 200 --seed 0
$PYTHON make_synthetic_pairs.py --layout stacked --num-pairs 200 --seed 0

echo "[2/5] Building train_synth15.jsonl (real train + 15% synthetic)"
$PYTHON make_mixed_train.py --ratio 0.15 --seed 42 \
    --synthetic-file data/synthetic/pairs_side_by_side.jsonl \
    --synthetic-file data/synthetic/pairs_stacked.jsonl

if [ "$SKIP_TRAIN" != "1" ]; then
    echo "[3/5] Training baseline (real data only)"
    $PYTHON -u train.py --config configs/default.yaml \
        --training-config configs/experiments/baseline_real_only.yaml 2>&1 | tee -a outputs/train_baseline.log

    echo "[4/5] Training synth15 (real + synthetic)"
    $PYTHON -u train.py --config configs/default.yaml \
        --training-config configs/experiments/synth15.yaml 2>&1 | tee -a outputs/train_synth15.log
fi

echo "[5/5] Evaluating both adapters on the same test set"
mkdir -p outputs/eval_results/baseline outputs/eval_results/synth15
$PYTHON evaluate.py --config configs/default.yaml --test-file "$TEST_FILE" \
    --adapter-path outputs/checkpoints/baseline/final_adapter \
    --output-report outputs/eval_results/baseline/report.json \
    --save-predictions outputs/eval_results/baseline/predictions.jsonl
$PYTHON evaluate.py --config configs/default.yaml --test-file "$TEST_FILE" \
    --adapter-path outputs/checkpoints/synth15/final_adapter \
    --output-report outputs/eval_results/synth15/report.json \
    --save-predictions outputs/eval_results/synth15/predictions.jsonl

$PYTHON -c "
import json
a=json.load(open('outputs/eval_results/baseline/report.json'))
b=json.load(open('outputs/eval_results/synth15/report.json'))
print(f\"{'metric':30s} {'baseline':>12s} {'synth15':>12s} {'delta':>10s}\")
for k in a:
    if isinstance(a[k], (int, float)) and k in b:
        print(f'{k:30s} {a[k]:12.4f} {b[k]:12.4f} {b[k]-a[k]:+10.4f}')
"
