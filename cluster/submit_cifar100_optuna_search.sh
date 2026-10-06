#!/bin/bash -l
#SBATCH --job-name=c100_optuna
#SBATCH --output=logs/cifar100_optuna_%A_%a.out
#SBATCH --error=logs/cifar100_optuna_%A_%a.err
#SBATCH --array=0-23%2
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ps
#SBATCH --exclude=antenna2
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

echo "CIFAR-100 controlled search"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "========================================"

which python
python -V

echo "----- NVIDIA -----"
nvidia-smi || true

echo "----- PyTorch -----"
python - <<'PY'
import sys
import optuna
import torch

print("Python executable:", sys.executable)
print("Optuna:", optuna.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available in this SLURM job."
    )

print("GPU count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "========================================"
echo "Resolving experiment case"
echo "========================================"

read -r DATASET CASE ALGORITHM < <(
    python run_case_index.py \
        --dataset cifar100 \
        --index "${SLURM_ARRAY_TASK_ID}"
)

echo "Dataset   : ${DATASET}"
echo "Case      : ${CASE}"
echo "Algorithm : ${ALGORITHM}"
echo "Model     : ResNet20"

echo "========================================"
echo "Starting Optuna search"
echo "========================================"

python run_optuna_dirichlet_search.py \
    --dataset "$DATASET" \
    --case "$CASE" \
    --algorithm "$ALGORITHM" \
    --search-config configs/controlled_search_grid.yaml