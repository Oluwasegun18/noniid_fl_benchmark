#!/bin/bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load python/3.12.0 2>/dev/null || true
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p logs data search_outputs partition_cache

python show_search_grid.py
python run_dirichlet_search.py --list
