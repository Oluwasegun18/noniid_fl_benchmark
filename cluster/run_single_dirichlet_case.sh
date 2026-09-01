#!/bin/bash -l
#SBATCH --job-name=fl_grid_one
#SBATCH --output=logs/grid_%j.out
#SBATCH --error=logs/grid_%j.err
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# -----------------------------------------------------------------------------
# Run one dataset / heterogeneity case.
#
# Examples:
#   sbatch --export=ALL,DATASET=cifar10,CASE=high cluster/run_single_dirichlet_case.sh
#   sbatch --export=ALL,DATASET=cifar10,CASE=mild cluster/run_single_dirichlet_case.sh
#   sbatch --export=ALL,DATASET=cifar10,CASE=iid  cluster/run_single_dirichlet_case.sh
#
# Valid DATASET values:
#   cifar10, cifar100, covtype, femnist, shakespeare
#
# Valid CASE values:
#   high  -> Dirichlet alpha=0.1
#   mild  -> Dirichlet alpha=0.5
#   iid   -> Dirichlet alpha=100 (IID-like)
# -----------------------------------------------------------------------------

module purge
module load python/3.12.0/default
module load cuda/12.8/default

DATASET="${DATASET:-cifar10}"
CASE="${CASE:-high}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

cd "${PROJECT_DIR}"
mkdir -p logs search_outputs partition_cache

printf '\n===== SLURM JOB INFORMATION =====\n'
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Dataset: ${DATASET}"
echo "Case: ${CASE}"
echo "Start time: $(date)"

printf '\n===== PYTHON / GPU CHECK =====\n'
echo "Python path:"
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

printf '\n===== STARTING CONTROLLED SEARCH =====\n'
python run_dirichlet_search.py \
    --dataset "${DATASET}" \
    --case "${CASE}" \
    --search-config configs/controlled_search_grid.yaml

printf '\n===== JOB COMPLETE =====\n'
echo "End time: $(date)"
