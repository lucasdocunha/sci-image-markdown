#!/bin/bash
#SBATCH --job-name=synth15
#SBATCH --partition=gpu            # TODO: nome da partição com GPU no seu cluster
#SBATCH --gres=gpu:1               # 1 GPU (QLoRA 4-bit cabe em <7 GB de VRAM)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00            # 2 treinos x 3 épocas + 2 avaliações; ajuste se a fila exigir
#SBATCH --output=outputs/slurm-%x-%j.out
#SBATCH --error=outputs/slurm-%x-%j.out
# Uncomment if your cluster requires an account:
##SBATCH --account=TODO_SUA_CONTA

# Submit from the repo root:  sbatch run_synth_slurm.sh
# outputs/ must exist before submission (SLURM does not create the log directory).
set -euo pipefail

REPO=/home/lucas/masters/sci-image-markdown
cd "$REPO"

echo "Job ${SLURM_JOB_ID} on $(hostname) started at $(date '+%F %T')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi indisponível')"

# Uncomment and adjust if your cluster exposes CUDA through modules:
# module load cuda/12.1

# Same pipeline as run_synth_experiment.sh: build pairs, build train_synth15.jsonl,
# train baseline and synth15, evaluate both on data/processed/test.jsonl.
bash "$REPO/run_synth_experiment.sh"

echo "Job finished at $(date '+%F %T')"
