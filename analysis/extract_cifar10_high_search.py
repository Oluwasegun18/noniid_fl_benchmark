from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter
from typing import Any

import optuna
import pandas as pd
import yaml

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState


# ============================================================
# Configuration
# ============================================================

SCENARIO = "cifar10_dirichlet_highly_non_iid_a0p01"

SEARCH_ROOT = Path("search_outputs") / SCENARIO

OUTPUT_ROOT = (
    Path("analysis_outputs")
    / "cifar10_highly_non_iid_search"
)

ALGORITHMS = [
    "fedavg",
    "fedprox",
    "scaffold",
    "fednova",
    "feddyn",
    "moon",
    "fedsam",
    "fedgucci",
]

EXPECTED_GRID_SIZE = {
    "fedavg": 36,
    "fedprox": 144,
    "scaffold": 36,
    "fednova": 36,
    "feddyn": 108,
    "moon": 144,
    "fedsam": 108,
    "fedgucci": 576,
}


# ============================================================
# Helpers
# ============================================================

def parameter_key(params: dict[str, Any]) -> str:
    """
    Canonical representation of one hyperparameter configuration.
    """
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def classify_failure(trial) -> str:
    """
    Normalize both old and new failure metadata.

    Important:
    - numerical divergence counts as an evaluated configuration
    - infrastructure interruption does NOT count as an evaluated result
    """

    failure_type = str(
        trial.user_attrs.get("failure_type", "")
    ).strip()

    failure_message = str(
        trial.user_attrs.get("failure_message", "")
    ).strip()

    error = str(
        trial.user_attrs.get("error", "")
    ).strip()

    combined = " ".join(
        [failure_type, failure_message, error]
    ).lower()

    if (
        failure_type == "diverged_nonfinite_loss"
        or "non-finite local loss" in combined
    ):
        return "diverged_nonfinite_loss"

    if failure_type == "infrastructure_interruption":
        return "infrastructure_interruption"

    if "out of memory" in combined:
        return "cuda_oom"

    if failure_type:
        return failure_type

    if error:
        return "other_error"

    if trial.state == TrialState.FAIL:
        return "unspecified_failure"

    return ""


def load_study(algorithm: str) -> optuna.Study:
    algorithm_root = SEARCH_ROOT / algorithm

    journal = algorithm_root / "optuna_journal.log"

    if not journal.exists():
        raise FileNotFoundError(
            f"Journal not found for {algorithm}: {journal}"
        )

    storage = JournalStorage(
        JournalFileBackend(str(journal.resolve()))
    )

    study_name = f"{SCENARIO}__{algorithm}"

    return optuna.load_study(
        study_name=study_name,
        storage=storage,
    )


def flatten_parameters(
    params: dict[str, Any],
) -> dict[str, Any]:

    result = {}

    for key, value in params.items():
        result[f"param__{key}"] = value

    return result


# ============================================================
# Extraction
# ============================================================

def extract_algorithm(
    algorithm: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:

    print(f"\n{'=' * 70}")
    print(f"Extracting {algorithm}")
    print(f"{'=' * 70}")

    study = load_study(algorithm)

    # --------------------------------------------------------
    # Trial-level records
    # --------------------------------------------------------

    trial_rows = []

    for trial in study.trials:

        failure_class = classify_failure(trial)

        row = {
            "scenario": SCENARIO,
            "algorithm": algorithm,
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective": trial.value,
            "configuration_key": parameter_key(
                dict(trial.params)
            ),
            "failure_type_normalized": failure_class,

            "run_status":
                trial.user_attrs.get("run_status"),

            "failure_type_raw":
                trial.user_attrs.get("failure_type"),

            "failure_message":
                trial.user_attrs.get("failure_message"),

            "error":
                trial.user_attrs.get("error"),

            "val_accuracy":
                trial.user_attrs.get("val_accuracy"),

            # Keep for audit only.
            # Do NOT use search-stage test accuracy for
            # final performance reporting.
            "test_accuracy_search_only":
                trial.user_attrs.get("test_accuracy"),

            "macro_f1":
                trial.user_attrs.get("macro_f1"),

            "mean_local_loss":
                trial.user_attrs.get("mean_local_loss"),

            "termination_round":
                trial.user_attrs.get("termination_round"),

            "stopping_reason":
                trial.user_attrs.get("stopping_reason"),

            "cumulative_wall_time_sec":
                trial.user_attrs.get(
                    "cumulative_wall_time_sec"
                ),

            "total_comm_bytes":
                trial.user_attrs.get("total_comm_bytes"),

            "run_gpu_energy_measured_j":
                trial.user_attrs.get(
                    "run_gpu_energy_measured_j"
                ),

            "run_total_energy_hybrid_j":
                trial.user_attrs.get(
                    "run_total_energy_hybrid_j"
                ),

            "output_dirs":
                trial.user_attrs.get("output_dirs"),

            "retry_of_trial":
                trial.user_attrs.get("retry_of_trial"),

            "retry_reason":
                trial.user_attrs.get("retry_reason"),
        }

        row.update(
            flatten_parameters(dict(trial.params))
        )

        trial_rows.append(row)

    # --------------------------------------------------------
    # Collapse trial records into UNIQUE configurations
    # --------------------------------------------------------

    by_config: dict[
        str,
        list[Any]
    ] = {}

    for trial in study.trials:

        if not trial.params:
            continue

        key = parameter_key(dict(trial.params))

        by_config.setdefault(
            key,
            [],
        ).append(trial)

    config_rows = []

    successful_keys = set()
    divergent_keys = set()
    infrastructure_only_keys = set()
    other_failed_keys = set()

    for key, trials in by_config.items():

        complete_trials = [
            t for t in trials
            if t.state == TrialState.COMPLETE
        ]

        failed_trials = [
            t for t in trials
            if t.state == TrialState.FAIL
        ]

        running_trials = [
            t for t in trials
            if t.state == TrialState.RUNNING
        ]

        waiting_trials = [
            t for t in trials
            if t.state == TrialState.WAITING
        ]

        failure_classes = [
            classify_failure(t)
            for t in failed_trials
        ]

        params = dict(trials[0].params)

        # --------------------------------------------
        # Determine scientific outcome
        # --------------------------------------------

        if complete_trials:

            outcome = "successful"
            successful_keys.add(key)

            # If retried and eventually successful,
            # use the strongest successful attempt.
            valid_complete = [
                t for t in complete_trials
                if t.value is not None
            ]

            if valid_complete:

                if study.direction.name == "MAXIMIZE":
                    representative = max(
                        valid_complete,
                        key=lambda t: t.value,
                    )
                else:
                    representative = min(
                        valid_complete,
                        key=lambda t: t.value,
                    )

            else:
                representative = complete_trials[0]

        elif (
            failed_trials
            and all(
                f == "infrastructure_interruption"
                for f in failure_classes
            )
        ):

            outcome = "infrastructure_interrupted"
            infrastructure_only_keys.add(key)
            representative = failed_trials[-1]

        elif (
            "diverged_nonfinite_loss"
            in failure_classes
        ):

            outcome = "numerical_divergence"
            divergent_keys.add(key)
            representative = failed_trials[-1]

        elif failed_trials:

            outcome = "other_failure"
            other_failed_keys.add(key)
            representative = failed_trials[-1]

        elif running_trials:

            outcome = "running"
            representative = running_trials[-1]

        elif waiting_trials:

            outcome = "waiting"
            representative = waiting_trials[-1]

        else:

            outcome = "unknown"
            representative = trials[-1]

        config_row = {
            "scenario": SCENARIO,
            "algorithm": algorithm,
            "configuration_key": key,
            "configuration_outcome": outcome,

            "number_of_trial_records":
                len(trials),

            "trial_numbers":
                "|".join(
                    str(t.number)
                    for t in trials
                ),

            "has_successful_trial":
                bool(complete_trials),

            "has_failed_trial":
                bool(failed_trials),

            "has_running_trial":
                bool(running_trials),

            "has_waiting_trial":
                bool(waiting_trials),

            "representative_trial":
                representative.number,

            "objective":
                representative.value,

            "val_accuracy":
                representative.user_attrs.get(
                    "val_accuracy"
                ),

            "macro_f1":
                representative.user_attrs.get(
                    "macro_f1"
                ),

            "mean_local_loss":
                representative.user_attrs.get(
                    "mean_local_loss"
                ),

            "termination_round":
                representative.user_attrs.get(
                    "termination_round"
                ),

            "stopping_reason":
                representative.user_attrs.get(
                    "stopping_reason"
                ),

            "total_comm_bytes":
                representative.user_attrs.get(
                    "total_comm_bytes"
                ),

            "failure_types":
                "|".join(
                    sorted(
                        set(
                            f
                            for f in failure_classes
                            if f
                        )
                    )
                ),
        }

        config_row.update(
            flatten_parameters(params)
        )

        config_rows.append(config_row)

    # --------------------------------------------------------
    # Selected configuration
    # --------------------------------------------------------

    complete = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.value is not None
    ]

    if complete:
        best = study.best_trial

        best_record = {
            "scenario": SCENARIO,
            "algorithm": algorithm,
            "best_trial_number": best.number,
            "best_validation_accuracy":
                best.user_attrs.get(
                    "val_accuracy",
                    best.value,
                ),
            "objective": best.value,
            "termination_round":
                best.user_attrs.get(
                    "termination_round"
                ),
            "stopping_reason":
                best.user_attrs.get(
                    "stopping_reason"
                ),
            "parameters_json":
                json.dumps(
                    best.params,
                    sort_keys=True,
                ),
        }

        best_record.update(
            flatten_parameters(dict(best.params))
        )

    else:
        best_record = {
            "scenario": SCENARIO,
            "algorithm": algorithm,
        }

    # --------------------------------------------------------
    # Algorithm-level summary
    # --------------------------------------------------------

    expected = EXPECTED_GRID_SIZE[algorithm]

    evaluated_keys = (
        successful_keys
        | divergent_keys
        | other_failed_keys
    )

    search_complete = (
        len(evaluated_keys) >= expected
    )

    divergence_rate = (
        100.0
        * len(divergent_keys)
        / expected
        if expected
        else None
    )

    success_rate = (
        100.0
        * len(successful_keys)
        / expected
        if expected
        else None
    )

    summary = {
        "scenario": SCENARIO,
        "algorithm": algorithm,

        "expected_grid_size": expected,

        "unique_configurations_seen":
            len(by_config),

        "unique_configurations_evaluated":
            len(evaluated_keys),

        "successful_unique_configurations":
            len(successful_keys),

        "numerically_divergent_unique_configurations":
            len(divergent_keys),

        "other_failed_unique_configurations":
            len(other_failed_keys),

        "infrastructure_interrupted_only":
            len(infrastructure_only_keys),

        "success_rate_percent":
            success_rate,

        "divergence_rate_percent":
            divergence_rate,

        "total_trial_records":
            len(study.trials),

        "complete_trial_records":
            sum(
                t.state == TrialState.COMPLETE
                for t in study.trials
            ),

        "failed_trial_records":
            sum(
                t.state == TrialState.FAIL
                for t in study.trials
            ),

        "running_trial_records":
            sum(
                t.state == TrialState.RUNNING
                for t in study.trials
            ),

        "waiting_trial_records":
            sum(
                t.state == TrialState.WAITING
                for t in study.trials
            ),

        "search_complete":
            search_complete,

        "best_trial_number":
            best_record.get("best_trial_number"),

        "best_validation_accuracy":
            best_record.get(
                "best_validation_accuracy"
            ),

        "best_parameters_json":
            best_record.get(
                "parameters_json"
            ),
    }

    print(
        f"Expected grid       : {expected}"
    )
    print(
        f"Evaluated unique    : "
        f"{len(evaluated_keys)}"
    )
    print(
        f"Successful unique   : "
        f"{len(successful_keys)}"
    )
    print(
        f"Numerical divergence: "
        f"{len(divergent_keys)}"
    )
    print(
        f"Infra interruption  : "
        f"{len(infrastructure_only_keys)}"
    )
    print(
        f"Best validation acc : "
        f"{summary['best_validation_accuracy']}"
    )
    print(
        f"Search complete     : "
        f"{search_complete}"
    )

    return (
        trial_rows,
        config_rows,
        summary,
        best_record,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_trials = []
    all_configs = []
    all_summaries = []
    all_best = []

    for algorithm in ALGORITHMS:

        (
            trials,
            configs,
            summary,
            best,
        ) = extract_algorithm(algorithm)

        all_trials.extend(trials)
        all_configs.extend(configs)
        all_summaries.append(summary)
        all_best.append(best)

    # --------------------------------------------------------
    # Save combined outputs
    # --------------------------------------------------------

    trials_df = pd.DataFrame(all_trials)

    configs_df = pd.DataFrame(all_configs)

    summary_df = pd.DataFrame(all_summaries)

    best_df = pd.DataFrame(all_best)

    trials_df.to_csv(
        OUTPUT_ROOT / "all_trial_records.csv",
        index=False,
    )

    configs_df.to_csv(
        OUTPUT_ROOT / "unique_configuration_results.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_ROOT / "algorithm_search_summary.csv",
        index=False,
    )

    best_df.to_csv(
        OUTPUT_ROOT / "selected_configurations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Also save selected parameters as YAML
    # --------------------------------------------------------

    selected_yaml = {}

    for record in all_best:

        algorithm = record["algorithm"]

        params = {}

        for key, value in record.items():

            if key.startswith("param__"):
                params[
                    key.replace(
                        "param__",
                        "",
                        1,
                    )
                ] = value

        selected_yaml[algorithm] = params

    (
        OUTPUT_ROOT
        / "selected_configurations.yaml"
    ).write_text(
        yaml.safe_dump(
            selected_yaml,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Simple manuscript-ready stability table
    # --------------------------------------------------------

    manuscript_columns = [
        "algorithm",
        "expected_grid_size",
        "successful_unique_configurations",
        "numerically_divergent_unique_configurations",
        "success_rate_percent",
        "divergence_rate_percent",
        "best_validation_accuracy",
        "best_trial_number",
    ]

    manuscript_df = (
        summary_df[manuscript_columns]
        .copy()
    )

    manuscript_df[
        "success_rate_percent"
    ] = manuscript_df[
        "success_rate_percent"
    ].round(2)

    manuscript_df[
        "divergence_rate_percent"
    ] = manuscript_df[
        "divergence_rate_percent"
    ].round(2)

    manuscript_df[
        "best_validation_accuracy"
    ] = manuscript_df[
        "best_validation_accuracy"
    ].round(6)

    manuscript_df.to_csv(
        OUTPUT_ROOT
        / "manuscript_search_table.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        "\nGenerated files:"
    )

    for path in sorted(
        OUTPUT_ROOT.iterdir()
    ):
        print(f"  {path.name}")

    print(
        "\nAlgorithm summary:"
    )

    print(
        manuscript_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()