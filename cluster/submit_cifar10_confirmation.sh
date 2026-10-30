#!/bin/bash -l
#SBATCH --job-name=c10_confirm
#SBATCH --output=logs/cifar10_confirm_%A_%a.out
#SBATCH --error=logs/cifar10_confirm_%A_%a.err
#SBATCH --array=8-23%1
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
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

echo "=================================================="
echo "CIFAR-10 confirmation"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=================================================="

echo "----- NVIDIA -----"
nvidia-smi

echo "----- Python / PyTorch -----"
which python
python -V

python - <<'PY'
import sys
import torch

print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: CUDA unavailable. Confirmation cannot continue."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

# Actual CUDA execution test
x = torch.tensor([1.0], device="cuda")
y = x * 2
torch.cuda.synchronize()

print("CUDA execution test:", y.item())
print("GPU check PASSED")
PY

echo "=================================================="
echo "Resolving confirmation case"
echo "=================================================="

read -r DATASET CASE ALGORITHM < <(
    python run_case_index.py \
        --dataset cifar10 \
        --index "${SLURM_ARRAY_TASK_ID}"
)

echo "Dataset   : ${DATASET}"
echo "Case      : ${CASE}"
echo "Algorithm : ${ALGORITHM}"

echo "=================================================="
echo "Starting confirmation"
echo "=================================================="

python run_confirmation.py \
    --dataset "$DATASET" \
    --case "$CASE" \
    --algorithm "$ALGORITHM" \
    --search-config configs/controlled_search_grid.yaml