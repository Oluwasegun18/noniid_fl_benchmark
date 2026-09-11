"""Resolve compact SLURM array indices for search/confirmation jobs."""
from __future__ import annotations

import argparse
from run_optuna_dirichlet_search import CASES, ALGORITHMS

# ALGORITHMS = [
#     "fedavg",
#     "fedprox",
#     "scaffold",
#     "fednova",
#     "feddyn",
#     "moon",
#     "fedsam",
#     "fedgucci",
# ]

# SYNTHETIC_CASES = [
#     "high",
#     "moderate",
#     "iid_like",
# ]

NATURAL_DATASETS = {
    "femnist",
    "shakespeare",
}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()
    dataset = args.dataset.lower()
    index = args.index

    if dataset in NATURAL_DATASETS:

        if not 0 <= index < len(ALGORITHMS):
            raise ValueError(
                f"{dataset} uses natural partitioning and has "
                f"{len(ALGORITHMS)} cases: indices 0-{len(ALGORITHMS)-1}."
            )

        algorithm = ALGORITHMS[index]

        print(
            dataset,
            "natural",
            algorithm,
        )

        return
    cases = list(CASES)
    total = len(cases) * len(ALGORITHMS)
    if not 0 <= args.index < total:
        raise SystemExit(f"index must be 0..{total-1}")
    case = cases[args.index // len(ALGORITHMS)]
    algorithm = ALGORITHMS[args.index % len(ALGORITHMS)]
    print(args.dataset, case, algorithm)

if __name__ == "__main__":
    main()
