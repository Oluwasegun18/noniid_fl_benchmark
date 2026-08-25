#!/bin/bash -l
#SBATCH --job-name=c10_dir_grid
#SBATCH --output=logs/cifar10_%A_%a.out
#SBATCH --error=logs/cifar10_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# -----------------------------------------------------------------------------
# CIFAR-10 only: run the three initial Dirichlet cases in parallel.
#   array task 0 -> highly non-IID, alpha=0.1
#   array task 1 -> mildly non-IID, alpha=0.5
#   array task 2 -> IID-like,       alpha=100
# -----------------------------------------------------------------------------

module purge
module load python/3.12.0/default
module load cuda/12.8/default

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
mkdir -p logs search_outputs partition_cache

CASES=(high mild iid)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"

printf '\n===== SLURM JOB INFORMATION =====\n'
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Dataset: cifar10"
echo "Case: ${CASE}"
echo "Start time: $(date)"

printf '\n===== PYTHON / GPU CHECK =====\n'
which python
python -V
nvidia-smi || true

python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(
        "ERROR: CUDA was requested, but PyTorch cannot use the GPU. "
        "Check the node driver, CUDA module, and PyTorch CUDA build."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY

python -c "import torchvision; print('torchvision:', torchvision.__version__, 'file:', torchvision.__file__)"

printf '\n===== STARTING CIFAR-10 SEARCH =====\n'
python run_dirichlet_search.py \
    --dataset cifar10 \
    --case "${CASE}" \
    --search-config configs/controlled_search_grid.yaml

printf '\n===== JOB COMPLETE =====\n'
echo "End time: $(date)"
