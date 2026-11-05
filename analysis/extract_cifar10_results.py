from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

ROOT = Path(".").resolve()
SEARCH_ROOT = ROOT / "search_outputs"
OUTPUT_DIR = ROOT / "extracted_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

SCENARIOS = {
    "cifar10_dirichlet_highly_non_iid_a0p01": {
        "label": "Highly Non-IID",
        "alpha": 0.01,
        "order": 0,
    },
    "cifar10_dirichlet_mildly_non_iid_a0p5": {
        "label": "Mildly Non-IID",
        "alpha": 0.5,
        "order": 1,
    },
    "cifar10_dirichlet_iid_like_a100p0": {
        "label": "IID-like",
        "alpha": 100.0,
        "order": 2,
    },
}


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return {}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        x = yaml.safe_load(path.read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return {}


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """
    Flatten nested YAML/JSON configuration dictionaries.

    Example:
        training:
          learning_rate: 0.1

    becomes:
        training.learning_rate = 0.1
    """
    out = {}

    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        else:
            out[name] = value

    return out


def safe_numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================
# SEARCH RESULTS
# ============================================================

def extract_search_results() -> pd.DataFrame:

    rows = []

    for scenario_name, meta in SCENARIOS.items():

        scenario_dir = SEARCH_ROOT / scenario_name

        if not scenario_dir.exists():
            print(f"[WARNING] Missing scenario: {scenario_dir}")
            continue

        for algorithm in ALGORITHMS:

            algorithm_dir = scenario_dir / algorithm

            if not algorithm_dir.exists():
                print(
                    f"[WARNING] Missing algorithm directory: "
                    f"{scenario_name}/{algorithm}"
                )
                continue

            summary_file = algorithm_dir / "search_summary.json"
            params_file = algorithm_dir / "best_parameters.yaml"
            config_file = algorithm_dir / "best_config.yaml"
            trials_file = algorithm_dir / "optuna_trials.csv"

            search_summary = load_json(summary_file)
            best_params = load_yaml(params_file)
            best_config = load_yaml(config_file)

            best_trial_number = search_summary.get(
                "best_trial_number"
            )

            best_val_accuracy = search_summary.get(
                "best_observed_validation_accuracy"
            )

            # ------------------------------------------------
            # Recover exact best-trial metrics from CSV
            # ------------------------------------------------

            best_trial_row = None

            if trials_file.exists():

                trials = pd.read_csv(trials_file)

                if "state" in trials.columns:
                    complete = trials[
                        trials["state"]
                        .astype(str)
                        .str.upper()
                        .eq("COMPLETE")
                    ].copy()
                else:
                    complete = trials.copy()

                if best_trial_number is not None:

                    trial_numbers = pd.to_numeric(
                        complete["trial_number"],
                        errors="coerce",
                    )

                    matched = complete[
                        trial_numbers == float(best_trial_number)
                    ]

                    if not matched.empty:
                        best_trial_row = matched.iloc[0]

                # Fallback if search_summary is incomplete
                if best_trial_row is None and not complete.empty:

                    metric_col = (
                        "objective"
                        if "objective" in complete.columns
                        else "val_accuracy"
                    )

                    complete[metric_col] = pd.to_numeric(
                        complete[metric_col],
                        errors="coerce",
                    )

                    valid = complete.dropna(subset=[metric_col])

                    if not valid.empty:
                        idx = valid[metric_col].idxmax()
                        best_trial_row = valid.loc[idx]

                        best_trial_number = int(
                            best_trial_row["trial_number"]
                        )

            # ------------------------------------------------
            # Extract trial metrics
            # ------------------------------------------------

            search_test_accuracy = np.nan
            search_macro_f1 = np.nan
            search_runtime_sec = np.nan
            search_comm_bytes = np.nan
            search_gpu_energy_j = np.nan
            search_termination_round = np.nan
            search_stopping_reason = None

            if best_trial_row is not None:

                if best_val_accuracy is None:
                    best_val_accuracy = best_trial_row.get(
                        "val_accuracy",
                        best_trial_row.get(
                            "objective",
                            np.nan,
                        ),
                    )

                search_test_accuracy = best_trial_row.get(
                    "test_accuracy",
                    np.nan,
                )

                search_macro_f1 = best_trial_row.get(
                    "macro_f1",
                    np.nan,
                )

                search_runtime_sec = best_trial_row.get(
                    "cumulative_wall_time_sec",
                    np.nan,
                )

                search_comm_bytes = best_trial_row.get(
                    "total_comm_bytes",
                    np.nan,
                )

                search_gpu_energy_j = best_trial_row.get(
                    "run_gpu_energy_measured_j",
                    np.nan,
                )

                search_termination_round = best_trial_row.get(
                    "termination_round",
                    np.nan,
                )

                search_stopping_reason = best_trial_row.get(
                    "stopping_reason",
                    None,
                )

            row = {
                "dataset": "cifar10",
                "scenario": scenario_name,
                "scenario_label": meta["label"],
                "dirichlet_alpha": meta["alpha"],
                "scenario_order": meta["order"],
                "algorithm": algorithm,
                "algorithm_order": ALGORITHMS.index(
                    algorithm
                ),
                "search_status": search_summary.get(
                    "status"
                ),
                "possible_configurations": search_summary.get(
                    "possible_configurations"
                ),
                "completed_trials": search_summary.get(
                    "completed_trials"
                ),
                "failed_trials": search_summary.get(
                    "failed_trials"
                ),
                "best_trial_number": best_trial_number,
                "best_val_accuracy": best_val_accuracy,
                "search_test_accuracy": search_test_accuracy,
                "search_macro_f1": search_macro_f1,
                "search_runtime_sec": search_runtime_sec,
                "search_comm_bytes": search_comm_bytes,
                "search_gpu_energy_j": search_gpu_energy_j,
                "search_termination_round":
                    search_termination_round,
                "search_stopping_reason":
                    search_stopping_reason,
            }

            # ---------------------------------------------
            # Put searched hyperparameters in columns
            # ---------------------------------------------

            flat_params = flatten_dict(best_params)

            for key, value in flat_params.items():
                row[f"param.{key}"] = value

            # Also keep selected useful resolved config
            flat_cfg = flatten_dict(best_config)

            useful_config_keys = [
                "training.learning_rate",
                "training.local_epochs",
                "training.batch_size",
                "algorithm.mu",
                "algorithm.alpha",
                "algorithm.rho",
                "algorithm.beta",
                "algorithm.num_anchors",
                "algorithm.anchor_count",
            ]

            for key in useful_config_keys:
                if key in flat_cfg:
                    row[key] = flat_cfg[key]

            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = (
            df.sort_values(
                [
                    "scenario_order",
                    "algorithm_order",
                ]
            )
            .reset_index(drop=True)
        )

    return df


# ============================================================
# CONFIRMATION RESULTS
# ============================================================

def extract_confirmation_results() -> pd.DataFrame:

    all_rows = []

    for scenario_name, meta in SCENARIOS.items():

        scenario_dir = SEARCH_ROOT / scenario_name

        for algorithm in ALGORITHMS:

            algorithm_dir = scenario_dir / algorithm

            csv_file = (
                algorithm_dir /
                "confirmation_results.csv"
            )

            if not csv_file.exists():
                print(
                    f"[WARNING] Missing confirmation: "
                    f"{scenario_name}/{algorithm}"
                )
                continue

            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                print(
                    f"[WARNING] Could not read "
                    f"{csv_file}: {exc}"
                )
                continue

            if df.empty:
                print(
                    f"[WARNING] Empty confirmation file: "
                    f"{csv_file}"
                )
                continue

            # Add identifiers ourselves rather than rely on
            # whatever naming convention the CSV uses.
            df["dataset"] = "cifar10"
            df["scenario"] = scenario_name
            df["scenario_label"] = meta["label"]
            df["dirichlet_alpha"] = meta["alpha"]
            df["scenario_order"] = meta["order"]
            df["algorithm"] = algorithm
            df["algorithm_order"] = (
                ALGORITHMS.index(algorithm)
            )

            df["source_file"] = str(csv_file)

            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(
        all_rows,
        ignore_index=True,
        sort=False,
    )

    result = result.sort_values(
        [
            "scenario_order",
            "algorithm_order",
        ]
    ).reset_index(drop=True)

    return result


# ============================================================
# INSPECT CONFIRMATION COLUMNS
# ============================================================

def print_confirmation_columns(df):

    if df.empty:
        return

    print("\nColumns in confirmation_results.csv:")
    for col in df.columns:
        print("  ", col)


# ============================================================
# CONFIRMATION AGGREGATION
# ============================================================

def aggregate_confirmation(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    # Support the names currently used by the benchmark.
    candidate_metrics = [
        "val_accuracy",
        "test_accuracy",
        "macro_f1",
        "mean_local_loss",
        "cumulative_wall_time_sec",
        "total_comm_bytes",
        "run_gpu_energy_measured_j",
        "run_total_energy_hybrid_j",
        "termination_round",
    ]

    metrics = [
        c for c in candidate_metrics
        if c in df.columns
    ]

    if not metrics:
        raise RuntimeError(
            "No expected confirmation metric columns found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    for col in metrics:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    group_cols = [
        "dataset",
        "scenario",
        "scenario_label",
        "dirichlet_alpha",
        "scenario_order",
        "algorithm",
        "algorithm_order",
    ]

    rows = []

    for keys, group in df.groupby(
        group_cols,
        dropna=False,
    ):

        row = dict(zip(group_cols, keys))

        # ---------------------------------------------
        # Number / list of confirmation seeds
        # ---------------------------------------------

        if "seed" in group.columns:

            seeds = (
                pd.to_numeric(
                    group["seed"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
            )

            seeds = sorted(seeds)

            row["n_confirmation_seeds"] = len(
                seeds
            )

            row["confirmation_seeds"] = ",".join(
                map(str, seeds)
            )

        else:
            row["n_confirmation_seeds"] = len(
                group
            )

        # ---------------------------------------------
        # Metric mean / std / min / max
        # ---------------------------------------------

        for metric in metrics:

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            row[f"{metric}_mean"] = (
                values.mean()
            )

            row[f"{metric}_std"] = (
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )

            row[f"{metric}_min"] = (
                values.min()
            )

            row[f"{metric}_max"] = (
                values.max()
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            [
                "scenario_order",
                "algorithm_order",
            ]
        ).reset_index(drop=True)

    return summary


# ============================================================
# MERGE SEARCH + CONFIRMATION
# ============================================================

def merge_search_confirmation(
    search_df,
    confirmation_summary,
):

    if search_df.empty:
        return confirmation_summary

    if confirmation_summary.empty:
        return search_df

    conf = confirmation_summary.copy()

    # Remove duplicate metadata before merge
    duplicate_cols = [
        "dataset",
        "scenario",
        "dirichlet_alpha",
        "scenario_order",
        "algorithm_order",
    ]

    for col in duplicate_cols:
        if col in conf.columns:
            conf = conf.drop(columns=col)

    merged = search_df.merge(
        conf,
        on=[
            "scenario_label",
            "algorithm",
        ],
        how="left",
    )

    merged = merged.sort_values(
        [
            "scenario_order",
            "algorithm_order",
        ]
    ).reset_index(drop=True)

    return merged


# ============================================================
# PAPER-READY TABLE
# ============================================================

def make_paper_table(df):

    if df.empty:
        return pd.DataFrame()

    rows = []

    for _, r in df.iterrows():

        row = {
            "Scenario":
                r.get("scenario_label"),

            "Algorithm":
                r.get("algorithm"),

            "Search Best Val. Accuracy":
                r.get("best_val_accuracy"),
        }

        # ---------------------------------------------
        # Hyperparameters
        # ---------------------------------------------

        parameter_mapping = {
            "Learning Rate":
                "training.learning_rate",

            "Local Epochs":
                "training.local_epochs",

            "Batch Size":
                "training.batch_size",

            "mu":
                "algorithm.mu",

            "alpha":
                "algorithm.alpha",

            "rho":
                "algorithm.rho",

            "beta":
                "algorithm.beta",

            "Anchors":
                "algorithm.num_anchors",
        }

        for display, source in parameter_mapping.items():

            if source in df.columns:
                row[display] = r.get(source)

        # ---------------------------------------------
        # Confirmation metrics
        # ---------------------------------------------

        pairs = {
            "Confirmation Test Accuracy":
                "test_accuracy",

            "Confirmation Macro-F1":
                "macro_f1",

            "Runtime (s)":
                "cumulative_wall_time_sec",

            "Communication (bytes)":
                "total_comm_bytes",

            "GPU Energy (J)":
                "run_gpu_energy_measured_j",

            "Termination Round":
                "termination_round",
        }

        for display, metric in pairs.items():

            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col not in df.columns:
                continue

            mean = r.get(mean_col)
            std = r.get(std_col)

            if pd.isna(mean):
                row[display] = ""
                continue

            if pd.notna(std):
                row[display] = (
                    f"{mean:.4f} ± {std:.4f}"
                )
            else:
                row[display] = f"{mean:.4f}"

        row["Confirmation Seeds"] = r.get(
            "confirmation_seeds"
        )

        row["N Seeds"] = r.get(
            "n_confirmation_seeds"
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Completeness check
# ============================================================

def print_completeness(
    search_df,
    confirmation_summary,
):

    print("\n" + "=" * 70)
    print("COMPLETENESS CHECK")
    print("=" * 70)

    expected = (
        len(SCENARIOS)
        * len(ALGORITHMS)
    )

    print(
        f"Expected scenario/algorithm pairs: "
        f"{expected}"
    )

    print(
        f"Search results found: "
        f"{len(search_df)}"
    )

    print(
        f"Confirmation summaries found: "
        f"{len(confirmation_summary)}"
    )

    print("\nPer scenario:")

    for scenario_name, meta in SCENARIOS.items():

        label = meta["label"]

        search_algos = set(
            search_df.loc[
                search_df["scenario_label"] == label,
                "algorithm",
            ]
        )

        if not confirmation_summary.empty:

            confirm_algos = set(
                confirmation_summary.loc[
                    confirmation_summary[
                        "scenario_label"
                    ] == label,
                    "algorithm",
                ]
            )

        else:
            confirm_algos = set()

        missing_search = (
            set(ALGORITHMS)
            - search_algos
        )

        missing_confirmation = (
            set(ALGORITHMS)
            - confirm_algos
        )

        print(f"\n{label}")

        print(
            "  Search:",
            len(search_algos),
            "/",
            len(ALGORITHMS),
        )

        print(
            "  Confirmation:",
            len(confirm_algos),
            "/",
            len(ALGORITHMS),
        )

        if missing_search:
            print(
                "  Missing search:",
                sorted(missing_search),
            )

        if missing_confirmation:
            print(
                "  Missing confirmation:",
                sorted(missing_confirmation),
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CIFAR-10 SEARCH + CONFIRMATION EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search_df = extract_search_results()

    search_file = (
        OUTPUT_DIR /
        "cifar10_search_results.csv"
    )

    search_df.to_csv(
        search_file,
        index=False,
    )

    print(
        f"\nSaved search results:\n"
        f"  {search_file}"
    )

    # --------------------------------------------------------
    # Confirmation seed-level
    # --------------------------------------------------------

    confirmation_df = (
        extract_confirmation_results()
    )

    print_confirmation_columns(
        confirmation_df
    )

    confirmation_file = (
        OUTPUT_DIR /
        "cifar10_confirmation_results.csv"
    )

    confirmation_df.to_csv(
        confirmation_file,
        index=False,
    )

    print(
        f"\nSaved confirmation seed results:\n"
        f"  {confirmation_file}"
    )

    # --------------------------------------------------------
    # Confirmation aggregate
    # --------------------------------------------------------

    confirmation_summary = (
        aggregate_confirmation(
            confirmation_df
        )
    )

    confirmation_summary_file = (
        OUTPUT_DIR /
        "cifar10_confirmation_summary.csv"
    )

    confirmation_summary.to_csv(
        confirmation_summary_file,
        index=False,
    )

    print(
        f"\nSaved confirmation summary:\n"
        f"  {confirmation_summary_file}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = merge_search_confirmation(
        search_df,
        confirmation_summary,
    )

    merged_file = (
        OUTPUT_DIR /
        "cifar10_search_confirmation_merged.csv"
    )

    merged.to_csv(
        merged_file,
        index=False,
    )

    print(
        f"\nSaved merged results:\n"
        f"  {merged_file}"
    )

    # --------------------------------------------------------
    # Paper table
    # --------------------------------------------------------

    paper = make_paper_table(
        merged
    )

    paper_file = (
        OUTPUT_DIR /
        "cifar10_paper_table.csv"
    )

    paper.to_csv(
        paper_file,
        index=False,
    )

    print(
        f"\nSaved paper-ready table:\n"
        f"  {paper_file}"
    )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    print_completeness(
        search_df,
        confirmation_summary,
    )

    print("\n" + "=" * 70)
    print("PAPER TABLE")
    print("=" * 70)

    if not paper.empty:
        print(
            paper.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()