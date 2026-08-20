#!/bin/bash
#SBATCH --job-name=fl-grid-one
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/grid_%j.out
#SBATCH --error=logs/grid_%j.err

set -euo pipefail

# Candidate and confirmation runs stop by validation convergence.
# The configured 1000 communication rounds are only a safety ceiling.
# If the SLURM wall-time is reached first, resubmit with the same output
# directory and resume enabled (or increase --time for the production run).

# Example:
# sbatch --export=ALL,DATASET=cifar10,CASE=high cluster/run_single_dirichlet_case.sh

DATASET="${DATASET:-cifar10}"
CASE="${CASE:-high}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

cd "${PROJECT_DIR}"
mkdir -p logs

if command -v module >/dev/null 2>&1; then
    module load python/3.12.0 2>/dev/null || true
fi

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

python run_dirichlet_search.py \
    --dataset "${DATASET}" \
    --case "${CASE}" \
    --search-config configs/controlled_search_grid.yaml
