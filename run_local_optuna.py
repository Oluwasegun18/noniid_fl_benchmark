from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, stdout_path=None, stderr_path=None):
    stdout_handle = None
    stderr_handle = None

    try:
        if stdout_path is not None:
            stdout_handle = open(stdout_path, "w", encoding="utf-8")

        if stderr_path is not None:
            stderr_handle = open(stderr_path, "w", encoding="utf-8")

        result = subprocess.run(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

        return result.returncode

    finally:
        if stdout_handle:
            stdout_handle.close()

        if stderr_handle:
            stderr_handle.close()


def resolve_case(dataset: str, index: int):
    result = subprocess.run(
        [
            sys.executable,
            "run_case_index.py",
            "--dataset",
            dataset,
            "--index",
            str(index),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    parts = result.stdout.strip().split()

    if len(parts) < 3:
        raise RuntimeError(
            f"Unexpected output from run_case_index.py: {result.stdout}"
        )

    return parts[0], parts[1], parts[2]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "cifar10",
            "cifar100",
            "covtype",
            "femnist",
            "shakespeare",
        ],
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--search-config",
        default="configs/controlled_search_grid.yaml",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    natural_datasets = {
        "femnist",
        "shakespeare",
    }

    if args.end_index is None:
        args.end_index = 7 if args.dataset in natural_datasets else 23

    project_dir = Path(__file__).resolve().parent
    log_dir = project_dir / "logs" / args.dataset

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Local Optuna runner")
    print("=" * 70)
    print(f"Python       : {sys.executable}")
    print(f"Dataset      : {args.dataset}")
    print(f"Case indices : {args.start_index}-{args.end_index}")
    print(f"Search config: {args.search_config}")
    print("=" * 70)

    for index in range(
        args.start_index,
        args.end_index + 1,
    ):
        try:
            dataset, case, algorithm = resolve_case(
                args.dataset,
                index,
            )

        except Exception as exc:
            print(
                f"[ERROR] Could not resolve case {index}: {exc}"
            )
            continue

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        out_file = (
            log_dir
            / f"{index:02d}_{algorithm}_{case}_{timestamp}.out"
        )

        err_file = (
            log_dir
            / f"{index:02d}_{algorithm}_{case}_{timestamp}.err"
        )

        print()
        print("=" * 70)
        print(
            f"Case {index}: "
            f"dataset={dataset}, "
            f"case={case}, "
            f"algorithm={algorithm}"
        )
        print(f"stdout: {out_file}")
        print(f"stderr: {err_file}")
        print("=" * 70)

        cmd = [
            sys.executable,
            "run_optuna_dirichlet_search.py",
            "--dataset",
            dataset,
            "--case",
            case,
            "--algorithm",
            algorithm,
            "--search-config",
            args.search_config,
        ]

        if args.n_trials is not None:
            cmd.extend(
                [
                    "--n-trials",
                    str(args.n_trials),
                ]
            )

        exit_code = run_command(
            cmd,
            stdout_path=out_file,
            stderr_path=err_file,
        )

        if exit_code == 0:
            print(
                f"[OK] Case {index} completed."
            )
        else:
            print(
                f"[FAILED] Case {index} exited with code {exit_code}."
            )
            print(
                f"See: {err_file}"
            )

    print()
    print("=" * 70)
    print("Local Optuna batch finished")
    print("=" * 70)


if __name__ == "__main__":
    main()