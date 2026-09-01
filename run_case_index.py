"""Resolve compact SLURM array indices for search/confirmation jobs."""
from __future__ import annotations

import argparse
from run_optuna_dirichlet_search import CASES, ALGORITHMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()
    cases = list(CASES)
    total = len(cases) * len(ALGORITHMS)
    if not 0 <= args.index < total:
        raise SystemExit(f"index must be 0..{total-1}")
    case = cases[args.index // len(ALGORITHMS)]
    algorithm = ALGORITHMS[args.index % len(ALGORITHMS)]
    print(args.dataset, case, algorithm)

if __name__ == "__main__":
    main()
