#!/bin/bash
#SBATCH --job-name=teleocr-expB
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --output=logs/slurm_teleocr_decomp_%j.out
#SBATCH --error=logs/slurm_teleocr_decomp_%j.err
#SBATCH --time=24:00:00

# Experimento B: pipeline de 3 modelos sobre TeleOCR.
#   [0] gera os JSONL decompostos (esqueleto, valores, fusão GT)
#   [1] M1 treino: imagem -> esqueleto
#   [2] M2 treino: imagem -> valores
#   [3] M3 treino: esqueleto + valores -> tabela final (só texto)
#   [4] avaliação end-to-end no conjunto de teste
#
# M1 e M2 são independentes: para rodá-los em paralelo, submeta dois jobs:
#   STAGES="1" sbatch scripts/slurm_teleocr_decomposed.sh
#   STAGES="2" sbatch scripts/slurm_teleocr_decomposed.sh
# Depois, com ambos prontos:  STAGES="0 3 4" sbatch scripts/slurm_teleocr_decomposed.sh
# (o 0 só é necessário se os JSONL ainda não existirem)

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

STAGES="${STAGES:-0 1 2 3 4}"
DATA_ROOT=/media/ssd2/datasets/sci-image-markdown-data
OUT=$DATA_ROOT/outputs
EVAL_DIR=$OUT/eval_results/teleocr_decomposed
mkdir -p "$EVAL_DIR" logs

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage 0; then
  echo "=== [0] Building decomposed JSONL datasets ==="
  .venv/bin/python -u scripts/build_decomposed_datasets.py --fusion-source gt 2>&1 | tee logs/build_decomposed.log
fi

if has_stage 1; then
  echo "=== [1] Training M1 (skeleton) ==="
  .venv/bin/python -u train.py --config configs/training/train_teleocr_skeleton.yaml 2>&1 | tee logs/train_teleocr_skeleton.log
fi

if has_stage 2; then
  echo "=== [2] Training M2 (values) ==="
  .venv/bin/python -u train.py --config configs/training/train_teleocr_values.yaml 2>&1 | tee logs/train_teleocr_values.log
fi

if has_stage 3; then
  echo "=== [3] Training M3 (fusion, text-only) ==="
  .venv/bin/python -u train.py --config configs/training/train_teleocr_fusion.yaml 2>&1 | tee logs/train_teleocr_fusion.log
fi

if has_stage 4; then
  echo "=== [4] End-to-end evaluation of the decomposed pipeline ==="
  .venv/bin/python -u evaluate_decomposed.py \
    --config-skeleton configs/training/train_teleocr_skeleton.yaml \
    --config-values configs/training/train_teleocr_values.yaml \
    --config-fusion configs/training/train_teleocr_fusion.yaml \
    --adapter-skeleton "$OUT/teleocr_decomp_skeleton/final_adapter" \
    --adapter-values "$OUT/teleocr_decomp_values/final_adapter" \
    --adapter-fusion "$OUT/teleocr_decomp_fusion/final_adapter" \
    --test-file data/processed/test.jsonl \
    --output-report "$EVAL_DIR/test_report.json" \
    --save-predictions "$EVAL_DIR/test_predictions.jsonl" 2>&1 | tee logs/eval_teleocr_decomposed.log
fi

echo "=== TeleOCR experiment B completed (stages: $STAGES) ==="
