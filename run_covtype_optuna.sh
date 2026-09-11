#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

DATASET="covtype"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-23}"
SEARCH_CONFIG="${SEARCH_CONFIG:-configs/controlled_search_grid.yaml}"

LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Local Optuna search"
echo "========================================"
echo "Dataset:       ${DATASET}"
echo "Cases:         ${START_INDEX}-${END_INDEX}"
echo "Project:       ${PROJECT_DIR}"
echo "Search config: ${SEARCH_CONFIG}"
echo "========================================"

which python
python -V

python - <<'PY'
import sys
import torch
import optuna

print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("Optuna:", optuna.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

for INDEX in $(seq "$START_INDEX" "$END_INDEX"); do

    echo
    echo "========================================"
    echo "Resolving case ${INDEX}"
    echo "========================================"

    read -r RESOLVED_DATASET CASE ALGORITHM < <(
        python run_case_index.py \
            --dataset "$DATASET" \
            --index "$INDEX"
    )

    echo "Dataset   : ${RESOLVED_DATASET}"
    echo "Case      : ${CASE}"
    echo "Algorithm : ${ALGORITHM}"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    OUT_FILE="${LOG_DIR}/${INDEX}_${ALGORITHM}_${CASE}_${TIMESTAMP}.out"
    ERR_FILE="${LOG_DIR}/${INDEX}_${ALGORITHM}_${CASE}_${TIMESTAMP}.err"

    echo "OUT: ${OUT_FILE}"
    echo "ERR: ${ERR_FILE}"

    set +e

    python run_optuna_dirichlet_search.py \
        --dataset "$RESOLVED_DATASET" \
        --case "$CASE" \
        --algorithm "$ALGORITHM" \
        --search-config "$SEARCH_CONFIG" \
        >"$OUT_FILE" \
        2>"$ERR_FILE"

    EXIT_CODE=$?

    set -e

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "WARNING: case ${INDEX} failed:"
        echo "  ${ALGORITHM} / ${CASE}"
        echo "  See ${ERR_FILE}"
        continue
    fi

    echo "Case ${INDEX} completed successfully."

done

echo
echo "========================================"
echo "Covtype search batch finished"
echo "========================================"