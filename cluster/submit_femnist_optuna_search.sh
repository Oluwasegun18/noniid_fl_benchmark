#!/bin/bash -l
#SBATCH --job-name=femnist_optuna
#SBATCH --output=logs/femnist_optuna_%A_%a.out
#SBATCH --error=logs/femnist_optuna_%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=ps
#SBATCH --exclude=antenna[1-3],xailab
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

echo "FEMNIST controlled search"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "========================================"
echo "----- Python environment -----"

which python
python -V
python -m pip --version

echo "----- NVIDIA -----"
nvidia-smi || true

echo "----- PyTorch / Optuna -----"
echo "----- Python dependencies -----"

python - <<'PY'
import sys
import optuna
import torch
import torchvision
import psutil
import yaml
import numpy
import pandas
import sklearn
import scipy

print("Python executable:", sys.executable)
print("Optuna:", optuna.__version__)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("psutil:", psutil.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("SciPy:", scipy.__version__)

print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available in this SLURM job."
    )

print("GPU count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

x = torch.tensor([1.0], device="cuda")
y = x * 2
torch.cuda.synchronize()

print("CUDA execution test:", y.item())
print("Environment check PASSED")
PY

echo "========================================"
echo "Resolving experiment case"
echo "========================================"

read -r DATASET CASE ALGORITHM < <(
    python run_case_index.py \
        --dataset femnist \
        --index "${SLURM_ARRAY_TASK_ID}"
)

echo "Dataset   : ${DATASET}"
echo "Case      : ${CASE}"
echo "Algorithm : ${ALGORITHM}"
echo "Partition : Natural FEMNIST client partition"

echo "========================================"
echo "Starting Optuna search"
echo "========================================"

python run_optuna_dirichlet_search.py \
    --dataset "$DATASET" \
    --case "$CASE" \
    --algorithm "$ALGORITHM" \
    --search-config configs/controlled_search_grid.yaml