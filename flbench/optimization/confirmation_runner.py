from __future__ import annotations

import csv
import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from flbench.experiment.runner import run_experiment

from .search_runner import _prepare_run_config


def _write_confirmation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "algorithm", "seed", "trial", "status", "test_accuracy", "macro_f1",
        "termination_round", "cumulative_wall_time_sec", "total_comm_bytes",
        "run_energy_primary_j", "run_gpu_energy_measured_j",
        "run_total_energy_hybrid_j", "run_energy_valid_for_ranking",
        "run_energy_invalid_reason", "run_gpu_counter_scope", "run_gpu_uuid",
        "run_concurrent_gpu_processes_detected", "output_dir", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_confirmation_from_dicts(
    base: dict[str, Any],
    search_cfg: dict[str, Any],
    algorithm: str,
    algorithm_root: Path,
    *,
    seeds: list[int] | None = None,
) -> Path:
    """Confirm an already-selected configuration with ranking-grade energy flags."""
    algorithm = str(algorithm).lower()
    algorithm_root = Path(algorithm_root)
    params_path = algorithm_root / "best_parameters.yaml"
    if not params_path.exists():
        raise FileNotFoundError(
            f"No selected parameters found at {params_path}. Run search first."
        )
    parameters = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}

    budget_cfg = search_cfg.get("budget", {})
    if seeds is None:
        seeds = [int(v) for v in budget_cfg.get("confirmation_seeds", [11, 22, 33])]
    termination_cfg = search_cfg.get("termination", {})
    rounds = int(
        termination_cfg.get("max_confirmation_rounds", budget_cfg.get("rounds_per_confirmation_run", 1000))
    )

    rows: list[dict[str, Any]] = []
    confirmation_root = algorithm_root / "confirmation_runs"
    confirmation_root.mkdir(parents=True, exist_ok=True)

    for trial_index, seed in enumerate(seeds, start=1):
        run_dir = confirmation_root / f"t{trial_index}_s{seed}"
        cfg = _prepare_run_config(
            deepcopy(base), algorithm, f"confirmation_t{trial_index}_s{seed}",
            parameters, int(seed), rounds, run_dir, termination_cfg,
        )
        cfg["experiment"]["trial"] = trial_index
        energy_cfg = cfg.setdefault("resources", {}).setdefault("energy", {})
        energy_cfg["role"] = "ranking"
        energy_cfg["require_valid_for_ranking"] = True
        energy_cfg.setdefault("check_gpu_isolation", True)
        energy_cfg.setdefault("gpu_warmup_steps", 2)

        try:
            output = Path(run_experiment(cfg))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "trial": trial_index,
                "status": "completed",
                "output_dir": str(output),
                **summary,
            })
        except Exception as exc:
            rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "trial": trial_index,
                "status": "failed",
                "output_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
            })

    (algorithm_root / "confirmation_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    _write_confirmation_csv(algorithm_root / "confirmation_results.csv", rows)

    completed = [r for r in rows if r.get("status") == "completed"]
    energy_valid = [
        r for r in completed
        if bool(r.get("run_energy_valid_for_ranking"))
        and r.get("run_energy_primary_j") is not None
    ]

    def stats(records: list[dict[str, Any]], key: str):
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if not values:
            return None, None
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    acc_mean, acc_std = stats(completed, "test_accuracy")
    runtime_mean, runtime_std = stats(completed, "cumulative_wall_time_sec")
    energy_mean, energy_std = stats(energy_valid, "run_energy_primary_j")
    hybrid_mean, hybrid_std = stats(completed, "run_total_energy_hybrid_j")

    summary = {
        "algorithm": algorithm,
        "status": "completed" if completed else "failed",
        "best_parameters": parameters,
        "confirmation_trials_requested": len(seeds),
        "confirmation_trials_completed": len(completed),
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "runtime_sec_mean": runtime_mean,
        "runtime_sec_std": runtime_std,
        "energy_ranking_policy": {
            "primary_metric": "measured GPU energy over full training run",
            "field": "run_energy_primary_j",
            "hybrid_energy_used_for_ranking": False,
            "search_energy_used_for_ranking": False,
        },
        "energy_valid_trials": len(energy_valid),
        "energy_invalid_trials": len(completed) - len(energy_valid),
        "gpu_energy_measured_j_mean_valid_only": energy_mean,
        "gpu_energy_measured_j_std_valid_only": energy_std,
        "hybrid_energy_j_mean_descriptive_only": hybrid_mean,
        "hybrid_energy_j_std_descriptive_only": hybrid_std,
        "energy_ranking_ready": len(energy_valid) == len(completed) and len(completed) > 0,
    }
    (algorithm_root / "confirmation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return algorithm_root
