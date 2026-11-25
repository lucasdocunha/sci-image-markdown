#!/bin/bash
#SBATCH --job-name=retomar-fila
#SBATCH --partition=dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/retomar_fila_%j.out
#SBATCH --error=logs/retomar_fila_%j.err

echo "=== Restoring Slurm queue after TeleOCR experiments ==="
scontrol release 447,448,449,453,454,455,456,457,458,459,461,462,463,464 || true
squeue
