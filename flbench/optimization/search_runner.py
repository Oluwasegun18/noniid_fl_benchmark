from __future__ import annotations

import csv
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from flbench.config import load_config, validate_config
from flbench.experiment.runner import run_experiment

from .budget import SearchBudget
from .search_space import apply_parameters, sample_candidates
from .selector import rank_candidates


CANDIDATE_FIELDS = [
    "candidate_id", "algorithm", "status", "rank", "seed", "trial",
    "val_accuracy", "test_accuracy", "macro_f1", "mean_local_loss",
    "cumulative_wall_time_sec", "total_comm_bytes",
    "round_compute_energy_reported_j", "error", "parameters_json",
    "output_dir",
]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CANDIDATE_FIELDS})


def _load_summary(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _prepare_run_config(
    base: dict[str, Any],
    algorithm: str,
    run_label: str,
    parameters: dict[str, Any],
    seed: int,
    rounds: int,
    output_dir: Path,
) -> dict[str, Any]:
    cfg = apply_parameters(base, parameters)
    cfg["algorithm"]["name"] = algorithm
    cfg["experiment"]["name"] = f"search_{algorithm}_{run_label}"
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["trial"] = 1

    # `output_dir` is already a unique candidate/seed directory. Instruct the
    # experiment runner not to append the long descriptive experiment ID.
    cfg["experiment"]["output_dir"] = str(output_dir)
    cfg["experiment"]["use_exact_output_dir"] = True

    cfg["federation"]["communication_rounds"] = int(rounds)
    # Search chooses only from validation results. Test values may still be
    # logged by the underlying runner, but they are never used by selector.py.
    validate_config(cfg)
    return cfg


def run_controlled_search(base_config_path: str, search_config_path: str) -> Path:
    base = load_config(base_config_path)
    search_cfg = _load_yaml(search_config_path)

    algorithms = list(search_cfg.get("algorithms", []))
    if not algorithms:
        algorithms = list(search_cfg.get("algorithm_search_spaces", {}).keys())
    if not algorithms:
        raise ValueError("No algorithms specified for controlled search.")

    budget_cfg = search_cfg["budget"]
    max_candidates = int(budget_cfg["max_configurations_per_algorithm"])
    search_seeds = [int(value) for value in budget_cfg.get("search_seeds", [1])]
    confirmation_seeds = [int(value) for value in budget_cfg.get("confirmation_seeds", [1, 2, 3])]
    search_rounds = int(budget_cfg["rounds_per_search_run"])
    confirmation_rounds = int(budget_cfg["rounds_per_confirmation_run"])
    max_runs = int(budget_cfg.get("max_total_runs_per_algorithm", max_candidates * len(search_seeds)))
    max_hours = budget_cfg.get("max_wall_time_hours_per_algorithm")
    sampler_seed = int(search_cfg.get("seeds", {}).get("search_sampler_seed", 2026))

    root = Path(search_cfg.get("output_dir", "search_outputs"))
    scenario = str(search_cfg.get("scenario_name", "controlled_search"))
    scenario_root = root / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)

    overall_summary: dict[str, Any] = {"scenario": scenario, "algorithms": {}}

    for algorithm_index, algorithm in enumerate(algorithms):
        algorithm_root = scenario_root / algorithm
        runs_root = algorithm_root / "candidate_runs"
        algorithm_root.mkdir(parents=True, exist_ok=True)

        candidates = sample_candidates(
            search_cfg,
            algorithm,
            max_candidates,
            sampler_seed + algorithm_index,
        )
        manifest = {
            "algorithm": algorithm,
            "candidate_count": len(candidates),
            "search_seeds": search_seeds,
            "confirmation_seeds": confirmation_seeds,
            "candidates": [
                {"candidate_id": f"c{index:04d}", "parameters": params}
                for index, params in enumerate(candidates, start=1)
            ],
        }
        (algorithm_root / "search_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        budget = SearchBudget(
            max_configurations=max_candidates,
            max_total_runs=max_runs,
            max_wall_time_hours=None if max_hours is None else float(max_hours),
        )
        records: list[dict[str, Any]] = []

        for candidate_index, parameters in enumerate(candidates, start=1):
            candidate_id = f"c{candidate_index:04d}"
            seed_summaries = []
            seed_output_dirs: list[str] = []
            error = None
            for seed in search_seeds:
                if not budget.can_start():
                    break

                # Keep the directory compact on Windows:
                # candidate_runs/c0001/s1/
                candidate_run_dir = runs_root / candidate_id / f"s{seed}"
                cfg = _prepare_run_config(
                    base,
                    algorithm,
                    f"{candidate_id}_s{seed}",
                    parameters,
                    seed,
                    search_rounds,
                    candidate_run_dir,
                )
                try:
                    output = run_experiment(cfg)
                    output_path = Path(output)
                    summary = _load_summary(output_path)
                    seed_summaries.append(summary)
                    seed_output_dirs.append(str(output_path))
                    budget.register("completed")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    seed_output_dirs.append(str(candidate_run_dir))
                    budget.register("failed")
                    break

            if seed_summaries:
                # Average the search metric across the allocated search seeds.
                numeric_keys = [
                    "val_accuracy", "test_accuracy", "macro_f1", "mean_local_loss",
                    "cumulative_wall_time_sec", "total_comm_bytes",
                    "round_compute_energy_reported_j",
                ]
                averaged = {}
                for key in numeric_keys:
                    values = [float(s[key]) for s in seed_summaries if s.get(key) is not None]
                    averaged[key] = sum(values) / len(values) if values else None
                status = "completed" if len(seed_summaries) == len(search_seeds) and error is None else "partial"
            else:
                averaged = {}
                status = "failed"

            records.append({
                "candidate_id": candidate_id,
                "algorithm": algorithm,
                "status": status,
                "rank": None,
                "seed": ",".join(map(str, search_seeds)),
                "trial": 1,
                **averaged,
                "error": error,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "output_dir": "|".join(seed_output_dirs),
            })
            _write_csv(algorithm_root / "candidate_results.csv", records)
            if not budget.can_start():
                break

        ranked = rank_candidates(records, search_cfg.get("selection", {}))
        _write_csv(algorithm_root / "candidate_results.csv", records)
        if not ranked:
            overall_summary["algorithms"][algorithm] = {
                "status": "no_valid_candidate", "budget": budget.to_dict()
            }
            continue

        best = ranked[0]
        best_params = json.loads(best["parameters_json"])
        best_cfg = apply_parameters(deepcopy(base), best_params)
        best_cfg["algorithm"]["name"] = algorithm
        (algorithm_root / "best_config.yaml").write_text(
            yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8"
        )
        (algorithm_root / "best_config.json").write_text(
            json.dumps(best_cfg, indent=2), encoding="utf-8"
        )

        confirmation_records = []
        confirmation_root = algorithm_root / "confirmation_runs"
        for trial, seed in enumerate(confirmation_seeds, start=1):
            confirmation_run_dir = confirmation_root / f"t{trial}_s{seed}"
            cfg = _prepare_run_config(
                base,
                algorithm,
                f"confirmation_t{trial}_s{seed}",
                best_params,
                seed,
                confirmation_rounds,
                confirmation_run_dir,
            )
            cfg["experiment"]["trial"] = trial
            try:
                output = run_experiment(cfg)
                output_path = Path(output)
                summary = _load_summary(output_path)
                confirmation_records.append({
                    "status": "completed",
                    "seed": seed,
                    "trial": trial,
                    "output_dir": str(output_path),
                    **summary,
                })
            except Exception as exc:
                confirmation_records.append({
                    "status": "failed",
                    "seed": seed,
                    "trial": trial,
                    "output_dir": str(confirmation_run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                })

        (algorithm_root / "confirmation_results.json").write_text(
            json.dumps(confirmation_records, indent=2), encoding="utf-8"
        )
        if confirmation_records:
            _write_csv(algorithm_root / "confirmation_results.csv", confirmation_records)
        completed_confirmation = [r for r in confirmation_records if r.get("status") == "completed"]
        confirmed_accuracy = None
        confirmed_std = None
        confirmation_runtime_mean = None
        confirmation_runtime_std = None
        if completed_confirmation:
            values = [float(r["test_accuracy"]) for r in completed_confirmation]
            confirmed_accuracy = statistics.mean(values)
            confirmed_std = statistics.stdev(values) if len(values) > 1 else 0.0
            runtimes = [float(r["cumulative_wall_time_sec"]) for r in completed_confirmation if r.get("cumulative_wall_time_sec") is not None]
            if runtimes:
                confirmation_runtime_mean = statistics.mean(runtimes)
                confirmation_runtime_std = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0

        algorithm_summary = {
            "status": "completed",
            "best_candidate_id": best["candidate_id"],
            "best_observed_validation_accuracy": best.get("val_accuracy"),
            "best_parameters": best_params,
            "confirmation_test_accuracy_mean": confirmed_accuracy,
            "confirmation_test_accuracy_std": confirmed_std,
            "confirmation_runtime_sec_mean": confirmation_runtime_mean,
            "confirmation_runtime_sec_std": confirmation_runtime_std,
            "confirmation_trials_completed": len(completed_confirmation),
            "search_budget": budget.to_dict(),
        }
        (algorithm_root / "search_summary.json").write_text(
            json.dumps(algorithm_summary, indent=2), encoding="utf-8"
        )
        overall_summary["algorithms"][algorithm] = algorithm_summary

    (scenario_root / "controlled_search_summary.json").write_text(
        json.dumps(overall_summary, indent=2), encoding="utf-8"
    )
    return scenario_root
