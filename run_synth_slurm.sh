#!/bin/bash
#SBATCH --job-name=synth15
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm-%x-%j.out
#SBATCH --error=outputs/slurm-%x-%j.out

# Submit from the repo root:  sbatch run_synth_slurm.sh
set -euo pipefail

REPO=/home/lucas.ocunha/sci-image-markdown
cd "$REPO"
mkdir -p outputs logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} on $(hostname) started at $(date '+%F %T')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi indisponível')"

# Uncomment and adjust if your cluster exposes CUDA through modules:
# module load cuda/12.1

# Same pipeline as run_synth_experiment.sh: build pairs, build train_synth15.jsonl,
# train baseline and synth15, evaluate both on data/processed/test.jsonl.
bash "$REPO/run_synth_experiment.sh"

echo "Job finished at $(date '+%F %T')"
