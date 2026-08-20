#!/bin/bash
#SBATCH --job-name=fl-dir-grid
#SBATCH --array=0-14%3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dirichlet_%A_%a.out
#SBATCH --error=logs/dirichlet_%A_%a.err

set -euo pipefail

# Candidate and confirmation runs stop by validation convergence.
# The configured 1000 communication rounds are only a safety ceiling.
# If the SLURM wall-time is reached first, resubmit with the same output
# directory and resume enabled (or increase --time for the production run).

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
    module load python/3.12.0 2>/dev/null || true
fi

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python run_dirichlet_search.py \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --search-config configs/controlled_search_grid.yaml
