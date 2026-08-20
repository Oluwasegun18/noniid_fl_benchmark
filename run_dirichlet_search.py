from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from flbench.config import load_config
from flbench.optimization.search_runner import (
    run_controlled_search_from_dicts,
)
from flbench.optimization.search_space import grid_candidates


DATASETS = {
    "cifar10": "configs/common_protocol.yaml",
    "cifar100": "configs/cifar100.yaml",
    "covtype": "configs/covtype.yaml",
    "femnist": "configs/femnist.yaml",
    "shakespeare": "configs/shakespeare.yaml",
}

CASES = {
    "high": {
        "alpha": 0.01,
        "label": "highly_non_iid",
    },
    "mild": {
        "alpha": 0.5,
        "label": "mildly_non_iid",
    },
    "iid": {
        "alpha": 100.0,
        "label": "iid_like",
    },
}

MATRIX = [
    (dataset, case)
    for dataset in DATASETS
    for case in CASES
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exhaustive controlled search for one "
            "dataset/Dirichlet-severity scenario."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument(
        "--index",
        type=int,
        help="Scenario matrix index 0-14 for SLURM arrays.",
    )
    parser.add_argument(
        "--search-config",
        default="configs/controlled_search_grid.yaml",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all scenario indices and exit.",
    )
    return parser.parse_args()


def resolve_selection(args):
    if args.index is not None:
        if not 0 <= args.index < len(MATRIX):
            raise ValueError(
                f"--index must be between 0 and {len(MATRIX)-1}."
            )
        return MATRIX[args.index]

    if args.dataset and args.case:
        return args.dataset, args.case

    raise ValueError(
        "Provide --index or both --dataset and --case."
    )


def main():
    args = parse_args()

    if args.list:
        for index, (dataset, case) in enumerate(MATRIX):
            print(
                f"{index:02d}: "
                f"{dataset:11s} "
                f"{case:5s} "
                f"alpha={CASES[case]['alpha']}"
            )
        return

    dataset, case = resolve_selection(args)
    case_info = CASES[case]

    base = load_config(DATASETS[dataset])
    search_cfg = yaml.safe_load(
        Path(args.search_config).read_text(encoding="utf-8")
    )

    base["data"]["dataset"] = dataset
    base["partition"]["method"] = "dirichlet"
    base["partition"]["alpha"] = case_info["alpha"]
    base["partition"]["cache_namespace"] = dataset
    base["experiment"]["name"] = (
        f"{dataset}_dirichlet_{case_info['label']}"
    )

    search_cfg = copy.deepcopy(search_cfg)
    alpha_tag = str(case_info["alpha"]).replace(".", "p")
    search_cfg["scenario_name"] = (
        f"{dataset}_dirichlet_"
        f"{case_info['label']}_a{alpha_tag}"
    )

    scenario_root = (
        Path(search_cfg.get("output_dir", "search_outputs"))
        / search_cfg["scenario_name"]
    )
    scenario_root.mkdir(parents=True, exist_ok=True)

    grid_sizes = {
        algorithm: len(grid_candidates(search_cfg, algorithm))
        for algorithm in search_cfg["algorithms"]
    }

    (scenario_root / "scenario_definition.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "case": case,
                "severity_label": case_info["label"],
                "partition_method": "dirichlet",
                "alpha": case_info["alpha"],
                "note": (
                    "alpha=100 is IID-like Dirichlet, not an "
                    "exact deterministic IID split."
                ),
                "base_config": base,
                "search_config": search_cfg,
                "grid_sizes": grid_sizes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Dataset: {dataset}")
    print(
        f"Case: {case_info['label']} | "
        f"Dirichlet alpha={case_info['alpha']}"
    )
    print("Algorithm grid sizes:")
    for algorithm, size in grid_sizes.items():
        print(f"  {algorithm:10s}: {size}")
    print(f"Total configurations: {sum(grid_sizes.values())}")

    output = run_controlled_search_from_dicts(
        base,
        search_cfg,
    )
    print(f"Completed scenario: {output}")


if __name__ == "__main__":
    main()
