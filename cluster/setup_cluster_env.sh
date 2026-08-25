#!/bin/bash -l

set -euo pipefail

# Optional environment/setup check.  Run this interactively on a login node
# before submitting the production jobs if the cluster Python module does not
# already contain all Python dependencies.

module purge
module load python/3.12.0/default
module load cuda/12.8/default

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs data search_outputs partition_cache

echo "Python: $(which python)"
python -V
nvidia-smi || true

# Install only missing Python packages into the user's site directory.  This
# avoids creating a virtual environment when the cluster module stack is
# already configured similarly to previous experiments.
python -m pip install --user -r requirements.txt

python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available on this node:", torch.cuda.is_available())
PY

python show_search_grid.py
python run_dirichlet_search.py --list
