#!/bin/bash
#SBATCH --job-name=eval-oficial
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=logs/aval_oficial_%x_%j.out
#SBATCH --error=logs/aval_oficial_%x_%j.err

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

source /home/lucas.ocunha/sci-image-markdown/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export HF_HOME=/media/ssd2/lucas.ocunha/.cache/huggingface

DATA_OUT=/media/ssd2/datasets/sci-image-markdown-data/outputs
OUT=data/resultados_pro_slide
META=data/raw/test_metadata.jsonl
mkdir -p "$OUT" logs

roda() {
  nome=$1; cfg=$2; adapter=$3; shift 3
  echo "================================================================="
  echo "🚀 Iniciando avaliacao: $nome"
  echo "Data/Hora: $(date)"
  echo "Node: $(hostname)"
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
  echo "Config: $cfg"
  echo "Adapter: $adapter"
  echo "Out Dir: $OUT/$nome"
  echo "================================================================="
  mkdir -p "$OUT/$nome"

  # 1) inferência no teste, salvando raw_output e id
  python -u evaluate.py --config "$cfg" "$@" \
    --adapter-path "$adapter" \
    --test-file data/processed/test.jsonl \
    --output-report "$OUT/$nome/test_report_local.json" \
    --save-predictions "$OUT/$nome/test_predictions.jsonl"

  # 2) nota oficial por painel
  python analyze_errors.py \
    --metadata "$META" \
    --repo-predictions "$OUT/$nome/test_predictions.jsonl" \
    --out-dir "$OUT/$nome"

  echo "================================================================="
  echo "✅ Concluido com sucesso: $nome"
  echo "Data/Hora: $(date)"
  if [ -f "$OUT/$nome/summary.json" ]; then
    echo "Summary:"
    cat "$OUT/$nome/summary.json"
  fi
  echo "================================================================="
}

TARGET=${1:-"all"}

case "$TARGET" in
  lightonocr_52_32)
    roda lightonocr_52_32 configs/training/train_lightonocr_3_4b.yaml "$DATA_OUT/lightonocr_3_4b_icdar_loss/final_adapter"
    ;;
  lightonocr_synth15)
    roda lightonocr_synth15 configs/training/train_lightonocr_3_4b_synth15.yaml "$DATA_OUT/lightonocr_3_4b_synth15_icdar_loss/final_adapter"
    ;;
  qwen7b)
    roda qwen7b configs/training/train_7b_icdar_loss.yaml "$DATA_OUT/qwen2_5_vl_7b_icdar_loss/final_adapter"
    ;;
  qwen3_vl_8b)
    roda qwen3_vl_8b configs/training/train_qwen3_vl_8b.yaml "$DATA_OUT/qwen3_vl_8b_icdar_loss/final_adapter"
    ;;
  qwen3b)
    roda qwen3b configs/default.yaml "$DATA_OUT/qwen2_5_vl_3b_orig_res/final_adapter" --training-config configs/evaluation/original_resolution.yaml
    ;;
  all)
    roda lightonocr_52_32 configs/training/train_lightonocr_3_4b.yaml "$DATA_OUT/lightonocr_3_4b_icdar_loss/final_adapter"
    roda lightonocr_synth15 configs/training/train_lightonocr_3_4b_synth15.yaml "$DATA_OUT/lightonocr_3_4b_synth15_icdar_loss/final_adapter"
    roda qwen7b configs/training/train_7b_icdar_loss.yaml "$DATA_OUT/qwen2_5_vl_7b_icdar_loss/final_adapter"
    roda qwen3_vl_8b configs/training/train_qwen3_vl_8b.yaml "$DATA_OUT/qwen3_vl_8b_icdar_loss/final_adapter"
    roda qwen3b configs/default.yaml "$DATA_OUT/qwen2_5_vl_3b_orig_res/final_adapter" --training-config configs/evaluation/original_resolution.yaml
    ;;
  *)
    echo "Alvo desconhecido: $TARGET"
    echo "Opcoes: lightonocr_52_32 | lightonocr_synth15 | qwen7b | qwen3_vl_8b | qwen3b | all"
    exit 1
    ;;
esac
