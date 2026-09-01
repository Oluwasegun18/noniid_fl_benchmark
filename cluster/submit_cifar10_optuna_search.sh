#!/bin/bash -l
#SBATCH --job-name=fl_optuna
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err
#SBATCH --array=0-23%4
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load python/3.12.0/default
module load cuda/12.8/default

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_DIR"

echo "========================================"
echo "Environment"
echo "========================================"

# IMPORTANT: create logs/ before sbatch because Slurm opens log paths before
# the job script starts: mkdir -p logs


which python
python -V

python - <<'PY'
import sys
import optuna
import torch

print("Python executable:", sys.executable)
print("Optuna:", optuna.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available in this SLURM job.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true

echo "========================================"
echo "Resolving experiment case"
echo "========================================"

read -r DATASET CASE ALGORITHM < <(
  python run_case_index.py --dataset cifar10 --index "${SLURM_ARRAY_TASK_ID}"
)

echo "Search case: dataset=${DATASET} case=${CASE} algorithm=${ALGORITHM}"

echo "========================================"
echo "Starting Optuna search"
echo "========================================"

# One SLURM task owns one GPU and executes one Optuna study. The %4 array cap
# uses at most 4 GPUs and 32 CPUs, matching the stated cluster quota.
python run_optuna_dirichlet_search.py \
  --dataset "$DATASET" \
  --case "$CASE" \
  --algorithm "$ALGORITHM" \
  --search-config configs/controlled_search_grid.yaml
