from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from flbench.config import load_config
from flbench.optimization.optuna_runner import run_optuna_search_from_dicts

DATASETS = {
    "cifar10": "configs/common_protocol.yaml",
    "cifar100": "configs/cifar100.yaml",
    "covtype": "configs/covtype.yaml",
    "femnist": "configs/femnist.yaml",
    "shakespeare": "configs/shakespeare.yaml",
}
CASES = {
    "high": {"alpha": 0.01, "label": "highly_non_iid"},
    "mild": {"alpha": 0.5, "label": "mildly_non_iid"},
    "iid": {"alpha": 100.0, "label": "iid_like"},
}
ALGORITHMS = ["fedavg", "fedprox", "scaffold", "fednova", "feddyn", "moon", "fedsam", "fedgucci"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one Optuna-controlled FL search case.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--search-config", default="configs/controlled_search_grid.yaml")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optional trials for this worker. Omit for a single worker to exhaust the grid.")
    return parser.parse_args()


def build_case(args):
    case = CASES[args.case]
    base = load_config(DATASETS[args.dataset])
    search_cfg = yaml.safe_load(Path(args.search_config).read_text(encoding="utf-8"))
    base["data"]["dataset"] = args.dataset
    base["partition"]["method"] = "dirichlet"
    base["partition"]["alpha"] = case["alpha"]
    base["partition"]["cache_namespace"] = args.dataset
    base["experiment"]["name"] = f"{args.dataset}_dirichlet_{case['label']}"
    search_cfg = copy.deepcopy(search_cfg)
    alpha_tag = str(case["alpha"]).replace(".", "p")
    search_cfg["scenario_name"] = f"{args.dataset}_dirichlet_{case['label']}_a{alpha_tag}"
    return base, search_cfg


def main():
    args = parse_args()
    base, search_cfg = build_case(args)
    scenario_root = Path(search_cfg.get("output_dir", "search_outputs")) / search_cfg["scenario_name"]
    scenario_root.mkdir(parents=True, exist_ok=True)
    definition = scenario_root / "scenario_definition.json"
    if not definition.exists():
        definition.write_text(json.dumps({
            "dataset": args.dataset,
            "case": args.case,
            "partition_method": "dirichlet",
            "alpha": CASES[args.case]["alpha"],
            "energy_policy": "search energy descriptive only; not used for ranking",
        }, indent=2), encoding="utf-8")
    out = run_optuna_search_from_dicts(base, search_cfg, args.algorithm, n_trials=args.n_trials)
    print(f"Optuna search worker completed: {out}")


if __name__ == "__main__":
    main()
