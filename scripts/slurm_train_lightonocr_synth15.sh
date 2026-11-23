#!/bin/bash
#SBATCH --job-name=tr-lighton-synth15
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --output=logs/slurm_train_lighton_synth15_%j.out
#SBATCH --error=logs/slurm_train_lighton_synth15_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export HF_HOME=/media/ssd2/lucas.ocunha/.cache/huggingface
export PYTHONPATH=.

mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/lightonocr_3_4b_synth15_icdar_loss
mkdir -p /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/lightonocr_3_4b_synth15_icdar_loss
mkdir -p logs

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date '+%F %T')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi indisponivel')"

echo "=== [1/2] Fine-Tuning LightOnOCR-3-4B (Loss v3 + Two-Table Synth15) ==="
.venv/bin/python -u train.py --config configs/training/train_lightonocr_3_4b_synth15.yaml 2>&1 | tee logs/train_lightonocr_3_4b_synth15.log

echo "=== [2/2] Running Evaluation on Test Set (373 samples) ==="
.venv/bin/python -u evaluate.py \
  --config configs/training/train_lightonocr_3_4b_synth15.yaml \
  --adapter-path /media/ssd2/datasets/sci-image-markdown-data/outputs/lightonocr_3_4b_synth15_icdar_loss/final_adapter \
  --test-file data/processed/test.jsonl \
  --output-report /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/lightonocr_3_4b_synth15_icdar_loss/test_report.json \
  --save-predictions /media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/lightonocr_3_4b_synth15_icdar_loss/test_predictions.jsonl 2>&1 | tee logs/eval_lightonocr_3_4b_synth15.log

echo "=== Evaluation Completed. Comparing with Previous Record (0.5232) ==="
.venv/bin/python -c "
import json
rec_path = '/media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/lightonocr_3_4b_icdar_loss_record_05232/test_report.json'
new_path = '/media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/lightonocr_3_4b_synth15_icdar_loss/test_report.json'
try:
    a = json.load(open(rec_path))
    b = json.load(open(new_path))
    print(f'{\"Metric\":30s} {\"Record (0.5232)\":>16s} {\"New (Synth15)\":>16s} {\"Delta\":>10s}')
    for k in b:
        v_a = a.get(k, 0)
        v_b = b.get(k, 0)
        delta = v_b - v_a
        print(f'{k:30s} {v_a:16.4f} {v_b:16.4f} {delta:+10.4f}')
except Exception as e:
    print('Could not generate diff table:', e)
"

echo "=== All Steps Finished Successfully at $(date '+%F %T') ==="
