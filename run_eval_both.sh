#!/bin/bash
set -euo pipefail

cd /home/lucas/masters/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PYTHON=".venv/bin/python"

echo "================================================================================"
echo "  Sci-Image-Markdown: Full Benchmark Evaluation (Base vs Fine-Tuned LoRA)      "
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')                                      "
echo "================================================================================"

mkdir -p outputs/eval_results/base_model outputs/eval_results/finetuned_model

# 1. Base Model (Zero-Shot Baseline)
echo ""
echo ">>> [1/2] Avaliando Modelo Base (Zero-Shot Baseline)..."
$PYTHON evaluate.py \
    --config configs/default.yaml \
    --test-file data/processed/test.jsonl \
    --output-report outputs/eval_results/base_model/report.json \
    --save-predictions outputs/eval_results/base_model/predictions.jsonl

# 2. Fine-Tuned Model (LoRA Adapter)
echo ""
echo ">>> [2/2] Avaliando Modelo Fine-Tuned (outputs/checkpoints/final_adapter)..."
$PYTHON evaluate.py \
    --config configs/default.yaml \
    --test-file data/processed/test.jsonl \
    --adapter-path outputs/checkpoints/final_adapter \
    --output-report outputs/eval_results/finetuned_model/report.json \
    --save-predictions outputs/eval_results/finetuned_model/predictions.jsonl

echo ""
echo "================================================================================"
echo "  Resumo Comparativo de Resultados (Base vs Fine-Tuned)                         "
echo "================================================================================"
$PYTHON -c "
import json
from rich.table import Table
from rich.console import Console

console = Console()
try:
    with open('outputs/eval_results/base_model/report.json') as f:
        base = json.load(f)
    with open('outputs/eval_results/finetuned_model/report.json') as f:
        ft = json.load(f)

    table = Table(title='Comparison: Base Model (Zero-Shot) vs Fine-Tuned LoRA', show_header=True, header_style='bold magenta')
    table.add_column('Metric', style='cyan')
    table.add_column('Base Model', style='yellow', justify='right')
    table.add_column('Fine-Tuned', style='green', justify='right')
    table.add_column('Delta (Gain)', style='bold white', justify='right')

    for k in base.keys():
        b_val = base.get(k, 0.0)
        f_val = ft.get(k, 0.0)
        diff = f_val - b_val if (isinstance(b_val, (int, float)) and isinstance(f_val, (int, float))) else 0.0
        delta_str = f'{diff:+.4f}' if diff != 0 else '0.0000'
        table.add_row(
            k,
            f'{b_val:.4f}' if isinstance(b_val, float) else str(b_val),
            f'{f_val:.4f}' if isinstance(f_val, float) else str(f_val),
            delta_str
        )
    console.print(table)
except Exception as e:
    print(f'Could not format comparison table: {e}')
"

echo ""
echo "================================================================================"
echo "  Benchmark finalizado com sucesso!                                            "
echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')                                     "
echo "================================================================================"
