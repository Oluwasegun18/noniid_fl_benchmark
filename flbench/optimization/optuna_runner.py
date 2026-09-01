from __future__ import annotations

import csv
import json
import os
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import yaml

from flbench.config import load_config
from flbench.experiment.runner import run_experiment

from .search_runner import _prepare_run_config
from .search_space import ParameterSpec, apply_parameters, effective_search_space, grid_candidates


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _scenario_root(search_cfg: dict[str, Any]) -> Path:
    return Path(search_cfg.get("output_dir", "search_outputs")) / str(
        search_cfg.get("scenario_name", "controlled_search")
    )


def _storage(search_cfg: dict[str, Any], algorithm_root: Path):
    """Build Optuna storage.

    For cluster workers, an explicit RDB URL is the most robust option. A
    Journal file is supported as a zero-service alternative on a shared
    filesystem. SQLite is kept only for single-worker/local use.
    """
    cfg = search_cfg.get("optuna", {}).get("storage", {})
    env_url = os.environ.get("OPTUNA_STORAGE_URL")
    url = env_url or cfg.get("url")
    if url:
        return str(url)

    backend = str(cfg.get("backend", "journal")).lower()
    if backend == "journal":
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        configured = cfg.get("path")
        path = Path(configured) if configured else algorithm_root / "optuna_journal.log"
        if not path.is_absolute():
            # Keep relative configured paths anchored to the scenario folder.
            path = algorithm_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return JournalStorage(JournalFileBackend(str(path)))

    if backend == "sqlite":
        db_path = algorithm_root / "optuna.db"
        return f"sqlite:///{db_path.resolve()}"

    raise ValueError(
        f"Unsupported Optuna storage backend '{backend}'. Use journal, sqlite, "
        "or provide optuna.storage.url / OPTUNA_STORAGE_URL."
    )


def _grid_space(search_cfg: dict[str, Any], algorithm: str) -> dict[str, list[Any]]:
    space = effective_search_space(search_cfg, algorithm)
    result: dict[str, list[Any]] = {}
    for name, spec in space.items():
        if spec.kind not in {"grid", "categorical"}:
            raise ValueError(
                "Optuna GridSampler requires explicit categorical/grid values; "
                f"'{name}' is {spec.kind}."
            )
        result[name] = list(spec.values)
    return result


def _sampler(search_cfg: dict[str, Any], algorithm: str, seed: int):
    sampler_name = str(
        search_cfg.get("optuna", {}).get(
            "sampler", search_cfg.get("search", {}).get("method", "grid")
        )
    ).lower()
    if sampler_name in {"grid", "exhaustive"}:
        return optuna.samplers.GridSampler(_grid_space(search_cfg, algorithm), seed=seed)
    if sampler_name in {"tpe", "bayesian"}:
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")


def _suggest(trial: optuna.Trial, name: str, spec: ParameterSpec) -> Any:
    if spec.kind in {"categorical", "grid"}:
        return trial.suggest_categorical(name, list(spec.values))
    if spec.kind == "uniform":
        return trial.suggest_float(name, float(spec.low), float(spec.high))
    if spec.kind == "log_uniform":
        return trial.suggest_float(name, float(spec.low), float(spec.high), log=True)
    if spec.kind == "int_uniform":
        return trial.suggest_int(name, int(spec.low), int(spec.high))
    raise ValueError(spec.kind)


def _selection_direction(search_cfg: dict[str, Any]) -> str:
    mode = str(search_cfg.get("selection", {}).get("mode", "maximize")).lower()
    return "maximize" if mode == "maximize" else "minimize"


def _metric_name(search_cfg: dict[str, Any]) -> str:
    return str(search_cfg.get("selection", {}).get("metric", "val_accuracy"))


def _export_study(
    study: optuna.Study,
    base: dict[str, Any],
    search_cfg: dict[str, Any],
    algorithm: str,
    algorithm_root: Path,
) -> None:
    """Export human-readable trial results and the current selected config."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective": trial.value,
            "parameters_json": json.dumps(trial.params, sort_keys=True),
        }
        for key, value in trial.user_attrs.items():
            row[key] = value
        rows.append(row)

    all_fields = [
        "trial_number", "state", "objective", "val_accuracy", "test_accuracy",
        "macro_f1", "mean_local_loss", "cumulative_wall_time_sec",
        "total_comm_bytes", "run_gpu_energy_measured_j",
        "run_total_energy_hybrid_j", "termination_round", "stopping_reason",
        "parameters_json", "output_dirs", "error",
    ]
    with (algorithm_root / "optuna_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in all_fields})

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return

    best = study.best_trial
    best_params = dict(best.params)
    best_cfg = apply_parameters(deepcopy(base), best_params)
    best_cfg["algorithm"]["name"] = algorithm
    (algorithm_root / "best_config.yaml").write_text(
        yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8"
    )
    (algorithm_root / "best_parameters.yaml").write_text(
        yaml.safe_dump(best_params, sort_keys=False), encoding="utf-8"
    )

    possible = None
    try:
        possible = len(grid_candidates(search_cfg, algorithm))
    except Exception:
        pass
    summary = {
        "status": "search_complete" if possible is not None and len(complete) >= possible else "search_in_progress",
        "study_name": study.study_name,
        "algorithm": algorithm,
        "sampler": type(study.sampler).__name__,
        "possible_configurations": possible,
        "completed_trials": len(complete),
        "failed_trials": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "best_trial_number": best.number,
        "best_observed_validation_accuracy": best.user_attrs.get("val_accuracy"),
        "best_parameters": best_params,
        "energy_policy": {
            "role": "descriptive",
            "ranking_use": False,
            "primary_logged_quantity": "run_gpu_energy_measured_j",
            "note": "Search-stage energy is reported as search cost only and is not used for algorithm energy ranking.",
        },
    }
    (algorithm_root / "search_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def run_optuna_search_from_dicts(
    base: dict[str, Any],
    search_cfg: dict[str, Any],
    algorithm: str,
    *,
    n_trials: int | None = None,
    study_name: str | None = None,
) -> Path:
    """Run/search one algorithm case using Optuna.

    The default GridSampler preserves the paper's exhaustive predefined search
    while providing resumability and shared-study worker coordination.
    """
    base = deepcopy(base)
    search_cfg = deepcopy(search_cfg)
    algorithm = str(algorithm).lower()

    algorithms = list(search_cfg.get("algorithms", []))
    if algorithms and algorithm not in algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not enabled in search config.")

    scenario_root = _scenario_root(search_cfg)
    algorithm_root = scenario_root / algorithm
    runs_root = algorithm_root / "candidate_runs"
    algorithm_root.mkdir(parents=True, exist_ok=True)

    budget_cfg = search_cfg.get("budget", {})
    search_seeds = [int(v) for v in budget_cfg.get("search_seeds", [1])]
    termination_cfg = search_cfg.get("termination", {})
    search_rounds = int(
        termination_cfg.get("max_search_rounds", budget_cfg.get("rounds_per_search_run", 1000))
    )
    sampler_seed = int(search_cfg.get("seeds", {}).get("search_sampler_seed", 2026))
    sampler_seed += sorted(search_cfg.get("algorithms", [algorithm])).index(algorithm) if algorithm in search_cfg.get("algorithms", []) else 0

    if study_name is None:
        study_name = f"{search_cfg.get('scenario_name', 'scenario')}__{algorithm}"

    study = optuna.create_study(
        study_name=study_name,
        storage=_storage(search_cfg, algorithm_root),
        sampler=_sampler(search_cfg, algorithm, sampler_seed),
        direction=_selection_direction(search_cfg),
        load_if_exists=True,
    )

    space = effective_search_space(search_cfg, algorithm)
    metric_name = _metric_name(search_cfg)

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            name: _suggest(trial, name, spec)
            for name, spec in sorted(space.items())
        }
        trial_dir = runs_root / f"trial_{trial.number:05d}"
        seed_summaries: list[dict[str, Any]] = []
        output_dirs: list[str] = []

        for seed in search_seeds:
            run_dir = trial_dir / f"s{seed}"
            cfg = _prepare_run_config(
                base, algorithm, f"trial_{trial.number:05d}_s{seed}", parameters,
                seed, search_rounds, run_dir, termination_cfg,
            )
            energy_cfg = cfg.setdefault("resources", {}).setdefault("energy", {})
            energy_cfg["role"] = "descriptive"
            energy_cfg["require_valid_for_ranking"] = False
            try:
                output = Path(run_experiment(cfg))
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                seed_summaries.append(summary)
                output_dirs.append(str(output))
            except Exception as exc:
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
                trial.set_user_attr("output_dirs", "|".join(output_dirs + [str(run_dir)]))
                raise

        def mean(key: str):
            values = [float(s[key]) for s in seed_summaries if s.get(key) is not None]
            return statistics.mean(values) if values else None

        attrs = {
            "val_accuracy": mean("val_accuracy"),
            "test_accuracy": mean("test_accuracy"),
            "macro_f1": mean("macro_f1"),
            "mean_local_loss": mean("mean_local_loss"),
            "cumulative_wall_time_sec": mean("cumulative_wall_time_sec"),
            "total_comm_bytes": mean("total_comm_bytes"),
            "run_gpu_energy_measured_j": mean("run_gpu_energy_measured_j"),
            "run_total_energy_hybrid_j": mean("run_total_energy_hybrid_j"),
            "termination_round": mean("termination_round"),
            "stopping_reason": "|".join(sorted({str(s.get('stopping_reason')) for s in seed_summaries})),
            "output_dirs": "|".join(output_dirs),
            "energy_role": "descriptive",
            "energy_valid_for_ranking": False,
        }
        for key, value in attrs.items():
            trial.set_user_attr(key, value)

        objective_value = attrs.get(metric_name)
        if objective_value is None:
            raise RuntimeError(f"Search metric '{metric_name}' unavailable for trial {trial.number}.")
        return float(objective_value)

    # With GridSampler, n_trials=None runs until the predefined grid is exhausted.
    # Multiple workers may point to the same study/storage; each worker should use
    # n_jobs=1 so SLURM owns GPU allocation.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(RuntimeError,))
    _export_study(study, base, search_cfg, algorithm, algorithm_root)
    return algorithm_root


def run_optuna_search(
    base_config_path: str,
    search_config_path: str,
    algorithm: str,
    *,
    n_trials: int | None = None,
) -> Path:
    return run_optuna_search_from_dicts(
        load_config(base_config_path), _load_yaml(search_config_path), algorithm,
        n_trials=n_trials,
    )
