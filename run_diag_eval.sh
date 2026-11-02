#!/bin/bash
#SBATCH --job-name=ev-diag
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=26G
#SBATCH --output=/home/adryan.silva/masters/sci-image-markdown/logs/diag_%j.out
#SBATCH --error=/home/adryan.silva/masters/sci-image-markdown/logs/diag_%j.err
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=adryanbrabuu@gmail.com

# usage: sbatch run_diag_eval.sh <ADAPTER_PATH> [DIAGNOSTIC_DIR]
# Evaluates one isolated-transform set at a time and prints the sensitivity ranking.
set -euo pipefail
ADAPTER=$1
DIAG=${2:-data/diagnostic}

cd $HOME/masters/sci-image-markdown
source .venv/bin/activate
export HF_HOME=$PWD/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/eval_results

# Loop only over the sets the repo actually wants (pruned DIAGNOSTIC_SETS) and
# resume automatically: sets whose report.json already exists are skipped.
WANTED=$(python -c "from augment_images import DIAGNOSTIC_SETS; print(' '.join(DIAGNOSTIC_SETS))")
for tag in $WANTED; do
    set_dir="$DIAG/$tag"
    if [ ! -d "$set_dir" ]; then
        echo "=== AVISO: pasta $set_dir ausente, pulando $tag ==="
        continue
    fi
    out="outputs/eval_results/diag_${tag}"
    if [ -f "$out/report.json" ]; then
        echo "=== $tag já avaliado (report.json existe), pulando ==="
        continue
    fi
    mkdir -p "$out"
    override=$(mktemp)
    cat > "$override" <<EOF
data:
  image_folder: "$set_dir"
EOF
    echo "=== eval diag [$tag] iniciado $(date) ==="
    python -u evaluate.py \
        --config configs/default.yaml \
        --training-config "$override" \
        --adapter-path "$ADAPTER" \
        --test-file "$set_dir/test.jsonl" \
        --output-report "$out/report.json" \
        --save-predictions "$out/predictions.jsonl" 2>&1 | tee "logs/eval_diag_${tag}.log"
    rm -f "$override"
done

echo "=== ranking de sensibilidade (icdar_score em %, delta vs baseline) ==="
python - <<'PY'
import json
from pathlib import Path

rows = []
for report in sorted(Path("outputs/eval_results").glob("diag_*/report.json")):
    tag = report.parent.name.replace("diag_", "")
    rows.append((tag, json.load(open(report))))
baseline = next((m for t, m in rows if t == "baseline"), None)


def pct(key, model):
    value = model.get(key)
    return f"{100 * value:.2f}" if isinstance(value, (int, float)) else "  -  "


print(f"{'set':18} {'icdar':>7} {'valid':>7} {'recall':>7} {'rmse':>9} {'delta_pp':>9}")
for tag, m in rows:
    delta = "" if baseline is None else f"{100 * (m['icdar_score'] - baseline['icdar_score']):+.2f}"
    rmse = m.get("cell_rmse")
    rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "-"
    print(
        f"{tag:18} {pct('icdar_score', m):>7} {pct('valid_table', m):>7} "
        f"{pct('cell_recall', m):>7} {rmse_str:>9} {delta:>9}"
    )
PY
echo "=== diagnostic eval finalizado $(date) ==="
