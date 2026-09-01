#!/bin/bash -l
#SBATCH --job-name=fl_opt_worker
#SBATCH --output=logs/optworker_%A_%a.out
#SBATCH --error=logs/optworker_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
module purge
module load python/3.12.0/default
module load cuda/12.8/default

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_DIR"

: "${DATASET:?Set DATASET, e.g. cifar10}"
: "${CASE:?Set CASE, e.g. high}"
: "${ALGORITHM:?Set ALGORITHM, e.g. fedgucci}"

# All workers point to the same Optuna study. For multi-node execution, set a
# robust shared RDB URL in OPTUNA_STORAGE_URL. Journal storage can be used when
# the cluster filesystem provides reliable file locking.
python run_optuna_dirichlet_search.py \
  --dataset "$DATASET" --case "$CASE" --algorithm "$ALGORITHM" \
  --search-config configs/controlled_search_grid.yaml
