#!/bin/bash -l
#SBATCH --job-name=fl_confirm
#SBATCH --output=logs/confirm_%A_%a.out
#SBATCH --error=logs/confirm_%A_%a.err
#SBATCH --array=0-23%1
#SBATCH --time=180:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load python/3.12.0/default
module load cuda/12.8/default

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_DIR"

read -r DATASET CASE ALGORITHM < <(
  python run_case_index.py --dataset cifar10 --index "${SLURM_ARRAY_TASK_ID}"
)

echo "Confirmation case: dataset=${DATASET} case=${CASE} algorithm=${ALGORITHM}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# Default %1 deliberately serializes confirmation cases to maximize energy
# measurement integrity. After verifying that NVML measurement is scoped to
# independently allocated GPUs/MIG instances, change %1 to %4 if desired.
python run_confirmation.py \
  --dataset "$DATASET" \
  --case "$CASE" \
  --algorithm "$ALGORITHM" \
  --search-config configs/controlled_search_grid.yaml
