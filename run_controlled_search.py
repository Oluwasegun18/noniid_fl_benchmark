from __future__ import annotations

import argparse

from flbench.optimization import run_controlled_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run equal-budget controlled hyperparameter search."
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--search-config", required=True)
    args = parser.parse_args()
    output = run_controlled_search(args.base_config, args.search_config)
    print(f"Controlled search completed: {output}")


if __name__ == "__main__":
    main()
