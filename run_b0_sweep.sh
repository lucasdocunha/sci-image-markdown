#!/bin/bash
# =============================================================================
# B0: resolution sweep with training and inference kept symmetric.
#
# Context: resolution used to be applied only on the inference side (a 280px
# PIL thumbnail in predictor.py), so the model trained at near-native resolution
# and was evaluated on thumbnails. It is now a single processor-level budget in
# visual tokens, shared by both paths.
#
# This sweep answers one question: how far does raising that budget improve
# Task 2, and where does it stop fitting in VRAM? Budgets run smallest first,
# so an out-of-memory failure at a large budget still leaves usable results.
#
#   100 tokens ~= 280x280 px  (the old inference-side behaviour; the baseline)
#   334 tokens ~= 512x512 px
#   752 tokens ~= 768x768 px
#  1337 tokens ~= 1024x1024 px
#
# Usage:
#   bash run_b0_sweep.sh                  # train + evaluate each budget
#   BUDGETS="100 334" bash run_b0_sweep.sh
#   EVAL_BASE=1 bash run_b0_sweep.sh      # also evaluate the zero-shot base model
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/default.yaml}"
TEST_FILE="${TEST_FILE:-data/processed/test.jsonl}"
# Sorted ascending: the OOM handler stops the sweep at the first failure, which
# only means "this is the VRAM ceiling" if larger budgets come later.
BUDGETS="$(echo "${BUDGETS:-100 334 752 1337}" | tr ' ' '\n' | sort -n | tr '\n' ' ')"
EVAL_BASE="${EVAL_BASE:-0}"
ROOT="outputs/b0_sweep"

mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.tsv"
[ -f "$SUMMARY" ] || printf 'budget\tapprox_px\tstage\tstatus\treport\n' > "$SUMMARY"

echo "================================================================================"
echo "  B0 resolution sweep"
echo "  budgets: $BUDGETS   |   started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

if ! $PYTHON -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "!! No CUDA device visible. 4-bit QLoRA requires an NVIDIA GPU."
    echo "!! Run this on the GPU host, not on a laptop without CUDA."
    exit 1
fi

record() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$SUMMARY"; }

for BUDGET in $BUDGETS; do
    APPROX_PX=$($PYTHON -c "print(int(($BUDGET*784)**0.5))")
    RUN="$ROOT/res_${BUDGET}"
    mkdir -p "$RUN"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo ">>> budget ${BUDGET} visual tokens (~${APPROX_PX}x${APPROX_PX} px)"
    echo "--------------------------------------------------------------------------------"

    # --- train -----------------------------------------------------------------
    if [ -d "$RUN/checkpoints/final_adapter" ]; then
        echo ">>> adapter already present, skipping training"
    else
        $PYTHON train.py \
            --config "$CONFIG" \
            -o "data.max_visual_tokens=${BUDGET}" \
            -o "training.output_dir=${RUN}/checkpoints" \
            -o "experiment_name=b0_res_${BUDGET}" \
            2>&1 | tee "$RUN/train.log"
        TRAIN_RC=${PIPESTATUS[0]}

        if [ "$TRAIN_RC" -ne 0 ]; then
            if grep -qiE "out of memory|CUDA out of memory" "$RUN/train.log"; then
                echo "!! OOM at budget ${BUDGET}. This is the VRAM ceiling; stopping the sweep."
                record "$BUDGET" "$APPROX_PX" train OOM -
                break
            fi
            echo "!! Training failed at budget ${BUDGET} (exit ${TRAIN_RC}); see $RUN/train.log"
            record "$BUDGET" "$APPROX_PX" train "FAILED(${TRAIN_RC})" -
            continue
        fi
        record "$BUDGET" "$APPROX_PX" train ok -
    fi

    # --- evaluate fine-tuned ---------------------------------------------------
    # Same budget on both sides: this is the symmetry the sweep is testing.
    $PYTHON evaluate.py \
        --config "$CONFIG" \
        --test-file "$TEST_FILE" \
        --adapter-path "$RUN/checkpoints/final_adapter" \
        --output-report "$RUN/report_finetuned.json" \
        --save-predictions "$RUN/predictions_finetuned.jsonl" \
        -o "data.max_visual_tokens=${BUDGET}" \
        2>&1 | tee "$RUN/eval.log"
    EVAL_RC=${PIPESTATUS[0]}
    [ "$EVAL_RC" -eq 0 ] \
        && record "$BUDGET" "$APPROX_PX" eval_finetuned ok "$RUN/report_finetuned.json" \
        || record "$BUDGET" "$APPROX_PX" eval_finetuned "FAILED(${EVAL_RC})" -

    # --- evaluate zero-shot base (optional) ------------------------------------
    if [ "$EVAL_BASE" = "1" ]; then
        $PYTHON evaluate.py \
            --config "$CONFIG" \
            --test-file "$TEST_FILE" \
            --output-report "$RUN/report_base.json" \
            --save-predictions "$RUN/predictions_base.jsonl" \
            -o "data.max_visual_tokens=${BUDGET}" \
            2>&1 | tee "$RUN/eval_base.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] \
            && record "$BUDGET" "$APPROX_PX" eval_base ok "$RUN/report_base.json" \
            || record "$BUDGET" "$APPROX_PX" eval_base FAILED -
    fi
done

echo ""
echo "================================================================================"
echo "  Sweep finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
$PYTHON scripts/summarize_sweep.py "$ROOT" || cat "$SUMMARY"
