from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from flbench.config import load_config
from flbench.optimization.confirmation_runner import run_confirmation_from_dicts
from run_optuna_dirichlet_search import DATASETS, CASES, ALGORITHMS


def main():
    parser = argparse.ArgumentParser(description="Run measurement-controlled confirmation for one selected case.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--search-config", default="configs/controlled_search_grid.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    case = CASES[args.case]
    base = load_config(DATASETS[args.dataset])
    base["data"]["dataset"] = args.dataset
    base["partition"]["method"] = "dirichlet"
    base["partition"]["alpha"] = case["alpha"]
    base["partition"]["cache_namespace"] = args.dataset

    search_cfg = yaml.safe_load(Path(args.search_config).read_text(encoding="utf-8"))
    alpha_tag = str(case["alpha"]).replace(".", "p")
    search_cfg["scenario_name"] = f"{args.dataset}_dirichlet_{case['label']}_a{alpha_tag}"
    algorithm_root = Path(search_cfg.get("output_dir", "search_outputs")) / search_cfg["scenario_name"] / args.algorithm
    out = run_confirmation_from_dicts(base, search_cfg, args.algorithm, algorithm_root, seeds=args.seeds)
    print(f"Confirmation completed: {out}")


if __name__ == "__main__":
    main()
