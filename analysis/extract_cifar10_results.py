from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(".").resolve()
SEARCH_ROOT = PROJECT_ROOT / "search_outputs"

OUTPUT_DIR = PROJECT_ROOT / "extracted_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHM_ORDER = [
    "fedavg",
    "fedprox",
    "scaffold",
    "fednova",
    "feddyn",
    "moon",
    "fedsam",
    "fedgucci",
]

# Used only for sorting/pretty reporting.
def scenario_label(name: str) -> str:
    x = name.lower()

    if "highly_non_iid" in x or "a0p01" in x:
        return "Highly Non-IID"

    if "mild" in x or "a0p5" in x:
        return "Mild Non-IID"

    if "iid_like" in x or "a100" in x:
        return "IID-like"

    return name


def scenario_order(label: str) -> int:
    order = {
        "Highly Non-IID": 0,
        "Mild Non-IID": 1,
        "IID-like": 2,
    }
    return order.get(label, 99)


def algorithm_order(name: str) -> int:
    name = str(name).lower()
    try:
        return ALGORITHM_ORDER.index(name)
    except ValueError:
        return 99


def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] Cannot read {path}: {exc}")
        return {}


def read_yaml(path: Path):
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        print(f"[WARNING] Cannot read {path}: {exc}")
        return {}


# ============================================================
# 1. Extract search results
# ============================================================

def extract_search_results() -> pd.DataFrame:
    rows = []

    scenario_dirs = sorted(
        p for p in SEARCH_ROOT.glob("cifar10_*")
        if p.is_dir()
    )

    print("\nCIFAR-10 search scenarios found:")
    for p in scenario_dirs:
        print("  ", p)

    for scenario_dir in scenario_dirs:

        scenario = scenario_dir.name
        pretty_scenario = scenario_label(scenario)

        for algorithm_dir in sorted(scenario_dir.iterdir()):

            if not algorithm_dir.is_dir():
                continue

            algorithm = algorithm_dir.name.lower()

            if algorithm not in ALGORITHM_ORDER:
                continue

            summary_file = algorithm_dir / "search_summary.json"
            params_file = algorithm_dir / "best_parameters.yaml"
            trials_file = algorithm_dir / "optuna_trials.csv"

            summary = read_json(summary_file)
            best_params = read_yaml(params_file)

            # ------------------------------------------------
            # Prefer best trial information from search summary
            # ------------------------------------------------

            best_trial = summary.get("best_trial_number")
            best_val = summary.get("best_observed_validation_accuracy")

            best_trial_row = None

            if trials_file.exists():
                trials = pd.read_csv(trials_file)

                # Keep completed trials only
                if "state" in trials.columns:
                    completed = trials[
                        trials["state"].astype(str).str.upper() == "COMPLETE"
                    ].copy()
                else:
                    completed = trials.copy()

                # Recover best trial if summary is incomplete
                if best_trial is None and not completed.empty:
                    metric = (
                        "objective"
                        if "objective" in completed.columns
                        else "val_accuracy"
                    )

                    completed[metric] = pd.to_numeric(
                        completed[metric],
                        errors="coerce",
                    )

                    completed = completed.dropna(subset=[metric])

                    if not completed.empty:
                        idx = completed[metric].idxmax()
                        best_trial_row = completed.loc[idx]
                        best_trial = int(best_trial_row["trial_number"])

                elif best_trial is not None:
                    match = trials[
                        pd.to_numeric(
                            trials["trial_number"],
                            errors="coerce",
                        ) == float(best_trial)
                    ]

                    if not match.empty:
                        best_trial_row = match.iloc[0]

            # ------------------------------------------------
            # Recover metrics directly from best trial
            # ------------------------------------------------

            search_test_accuracy = np.nan
            macro_f1 = np.nan
            runtime = np.nan
            comm_bytes = np.nan
            gpu_energy = np.nan
            termination_round = np.nan
            stopping_reason = None

            if best_trial_row is not None:

                if pd.isna(best_val):
                    best_val = best_trial_row.get(
                        "val_accuracy",
                        best_trial_row.get("objective", np.nan),
                    )

                search_test_accuracy = best_trial_row.get(
                    "test_accuracy",
                    np.nan,
                )

                macro_f1 = best_trial_row.get(
                    "macro_f1",
                    np.nan,
                )

                runtime = best_trial_row.get(
                    "cumulative_wall_time_sec",
                    np.nan,
                )

                comm_bytes = best_trial_row.get(
                    "total_comm_bytes",
                    np.nan,
                )

                gpu_energy = best_trial_row.get(
                    "run_gpu_energy_measured_j",
                    np.nan,
                )

                termination_round = best_trial_row.get(
                    "termination_round",
                    np.nan,
                )

                stopping_reason = best_trial_row.get(
                    "stopping_reason",
                    None,
                )

            row = {
                "dataset": "cifar10",
                "scenario": scenario,
                "scenario_label": pretty_scenario,
                "algorithm": algorithm,
                "search_status": summary.get("status"),
                "best_trial": best_trial,
                "best_val_accuracy": best_val,
                "search_test_accuracy": search_test_accuracy,
                "search_macro_f1": macro_f1,
                "search_runtime_sec": runtime,
                "search_comm_bytes": comm_bytes,
                "search_gpu_energy_j": gpu_energy,
                "search_termination_round": termination_round,
                "search_stopping_reason": stopping_reason,
            }

            # Add hyperparameters as normal columns
            for key, value in best_params.items():
                row[key] = value

            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["_scenario_order"] = df["scenario_label"].map(
            scenario_order
        )
        df["_algorithm_order"] = df["algorithm"].map(
            algorithm_order
        )

        df = (
            df.sort_values(
                ["_scenario_order", "_algorithm_order"]
            )
            .drop(
                columns=[
                    "_scenario_order",
                    "_algorithm_order",
                ]
            )
            .reset_index(drop=True)
        )

    return df


# ============================================================
# 2. Find confirmation result files
# ============================================================

def find_confirmation_files():
    """
    Search the project recursively because confirmation
    output directory names may differ between versions.
    """

    files = []

    for p in PROJECT_ROOT.rglob("confirmation_results.csv"):

        # Avoid extracting our own output
        if OUTPUT_DIR in p.parents:
            continue

        files.append(p)

    return sorted(set(files))


# ============================================================
# 3. Extract confirmation results
# ============================================================

def extract_confirmation_results() -> pd.DataFrame:

    files = find_confirmation_files()

    print("\nConfirmation files discovered:")
    for p in files:
        print("  ", p)

    frames = []

    for path in files:

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARNING] Cannot read {path}: {exc}")
            continue

        if df.empty:
            continue

        # ----------------------------------------------
        # Determine whether this belongs to CIFAR-10
        # ----------------------------------------------

        is_cifar10 = "cifar10" in str(path).lower()

        if "dataset" in df.columns:
            dataset_match = (
                df["dataset"]
                .astype(str)
                .str.lower()
                .eq("cifar10")
            )

            if dataset_match.any():
                df = df[dataset_match].copy()
                is_cifar10 = True

        if not is_cifar10:
            continue

        if df.empty:
            continue

        df["source_file"] = str(path)

        frames.append(df)

    if not frames:
        print(
            "\n[WARNING] No CIFAR-10 confirmation_results.csv "
            "files were found."
        )
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # Remove accidental duplicates
    result = result.drop_duplicates().reset_index(drop=True)

    # ----------------------------------------------
    # Normalize common column names
    # ----------------------------------------------

    rename_candidates = {
        "method": "algorithm",
        "algo": "algorithm",
        "seed_value": "seed",
        "accuracy": "test_accuracy",
        "test_acc": "test_accuracy",
        "f1": "macro_f1",
        "runtime_sec": "cumulative_wall_time_sec",
        "wall_time_sec": "cumulative_wall_time_sec",
        "communication_bytes": "total_comm_bytes",
        "gpu_energy_j": "run_gpu_energy_measured_j",
    }

    for old, new in rename_candidates.items():
        if old in result.columns and new not in result.columns:
            result = result.rename(columns={old: new})

    if "algorithm" in result.columns:
        result["algorithm"] = (
            result["algorithm"]
            .astype(str)
            .str.lower()
        )

    # Try to infer scenario when missing
    if "scenario" not in result.columns:
        result["scenario"] = result["source_file"].apply(
            infer_scenario_from_path
        )

    if "scenario_label" not in result.columns:
        result["scenario_label"] = result["scenario"].apply(
            scenario_label
        )

    return result


def infer_scenario_from_path(path):
    x = str(path).lower()

    if "highly_non_iid" in x or "a0p01" in x:
        return "cifar10_dirichlet_highly_non_iid_a0p01"

    if "mild" in x or "a0p5" in x:
        return "cifar10_dirichlet_mild_non_iid_a0p5"

    if "iid_like" in x or "a100" in x:
        return "cifar10_dirichlet_iid_like_a100"

    return "unknown"


# ============================================================
# 4. Aggregate confirmation seeds
# ============================================================

def aggregate_confirmation(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    # Columns we want to summarize if available
    metrics = [
        "test_accuracy",
        "macro_f1",
        "cumulative_wall_time_sec",
        "total_comm_bytes",
        "run_gpu_energy_measured_j",
        "termination_round",
    ]

    metrics = [
        col for col in metrics
        if col in df.columns
    ]

    for col in metrics:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    group_cols = [
        col for col in
        ["dataset", "scenario", "scenario_label", "algorithm"]
        if col in df.columns
    ]

    if "algorithm" not in group_cols:
        raise ValueError(
            "Could not identify the algorithm column in "
            "confirmation_results.csv"
        )

    agg_spec = {}

    for metric in metrics:
        agg_spec[f"{metric}_mean"] = (
            metric,
            "mean",
        )
        agg_spec[f"{metric}_std"] = (
            metric,
            "std",
        )

    if "seed" in df.columns:
        agg_spec["confirmation_seeds"] = (
            "seed",
            lambda x: ",".join(
                map(
                    str,
                    sorted(
                        pd.Series(x)
                        .dropna()
                        .unique()
                    ),
                )
            ),
        )

        agg_spec["n_confirmation_seeds"] = (
            "seed",
            "nunique",
        )
    else:
        first_metric = metrics[0]

        agg_spec["n_confirmation_seeds"] = (
            first_metric,
            "count",
        )

    summary = (
        df.groupby(
            group_cols,
            dropna=False,
        )
        .agg(**agg_spec)
        .reset_index()
    )

    # Standard deviations are NaN for one seed -> make explicit
    std_cols = [
        c for c in summary.columns
        if c.endswith("_std")
    ]

    summary[std_cols] = summary[std_cols].fillna(0.0)

    return summary


# ============================================================
# 5. Merge search + confirmation
# ============================================================

def merge_results(
    search_df: pd.DataFrame,
    confirmation_summary: pd.DataFrame,
) -> pd.DataFrame:

    if search_df.empty:
        return confirmation_summary

    if confirmation_summary.empty:
        return search_df

    merge_cols = [
        "algorithm",
        "scenario_label",
    ]

    final = search_df.merge(
        confirmation_summary,
        on=merge_cols,
        how="left",
        suffixes=("_search", "_confirmation"),
    )

    final["_scenario_order"] = (
        final["scenario_label"]
        .map(scenario_order)
    )

    final["_algorithm_order"] = (
        final["algorithm"]
        .map(algorithm_order)
    )

    final = (
        final.sort_values(
            ["_scenario_order", "_algorithm_order"]
        )
        .drop(
            columns=[
                "_scenario_order",
                "_algorithm_order",
            ]
        )
        .reset_index(drop=True)
    )

    return final


# ============================================================
# 6. Paper-friendly table
# ============================================================

def make_paper_table(final_df: pd.DataFrame) -> pd.DataFrame:

    if final_df.empty:
        return final_df

    result = pd.DataFrame()

    result["Scenario"] = final_df["scenario_label"]
    result["Algorithm"] = final_df["algorithm"]

    if "best_val_accuracy" in final_df.columns:
        result["Search Val. Acc."] = final_df[
            "best_val_accuracy"
        ]

    acc_mean = "test_accuracy_mean"
    acc_std = "test_accuracy_std"

    if (
        acc_mean in final_df.columns
        and acc_std in final_df.columns
    ):
        result["Confirmation Accuracy"] = final_df.apply(
            lambda r: (
                f"{r[acc_mean]:.4f} ± "
                f"{r[acc_std]:.4f}"
                if pd.notna(r[acc_mean])
                else ""
            ),
            axis=1,
        )

    if (
        "macro_f1_mean" in final_df.columns
        and "macro_f1_std" in final_df.columns
    ):
        result["Macro-F1"] = final_df.apply(
            lambda r: (
                f"{r['macro_f1_mean']:.4f} ± "
                f"{r['macro_f1_std']:.4f}"
                if pd.notna(r["macro_f1_mean"])
                else ""
            ),
            axis=1,
        )

    # Hyperparameters
    parameter_columns = [
        "training.learning_rate",
        "training.local_epochs",
        "training.batch_size",
        "algorithm.mu",
        "algorithm.alpha",
        "algorithm.rho",
        "algorithm.beta",
        "algorithm.num_anchors",
    ]

    for col in parameter_columns:
        if col in final_df.columns:
            result[col] = final_df[col]

    if "cumulative_wall_time_sec_mean" in final_df.columns:
        result["Runtime (s)"] = final_df[
            "cumulative_wall_time_sec_mean"
        ]

    if "total_comm_bytes_mean" in final_df.columns:
        result["Communication (bytes)"] = final_df[
            "total_comm_bytes_mean"
        ]

    if "run_gpu_energy_measured_j_mean" in final_df.columns:
        result["GPU Energy (J)"] = final_df[
            "run_gpu_energy_measured_j_mean"
        ]

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EXTRACTING CIFAR-10 RESULTS")
    print("=" * 70)

    # --------------------------------------------
    # Search
    # --------------------------------------------

    search_df = extract_search_results()

    search_path = (
        OUTPUT_DIR /
        "cifar10_search_results.csv"
    )

    search_df.to_csv(
        search_path,
        index=False,
    )

    print(
        f"\nSearch results: {len(search_df)} rows"
    )
    print(f"Saved: {search_path}")

    # --------------------------------------------
    # Confirmation
    # --------------------------------------------

    confirmation_df = extract_confirmation_results()

    confirmation_path = (
        OUTPUT_DIR /
        "cifar10_confirmation_results.csv"
    )

    confirmation_df.to_csv(
        confirmation_path,
        index=False,
    )

    print(
        f"\nConfirmation seed-level results: "
        f"{len(confirmation_df)} rows"
    )
    print(f"Saved: {confirmation_path}")

    # --------------------------------------------
    # Aggregate confirmation
    # --------------------------------------------

    confirmation_summary = aggregate_confirmation(
        confirmation_df
    )

    summary_path = (
        OUTPUT_DIR /
        "cifar10_confirmation_summary.csv"
    )

    confirmation_summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nConfirmation summary: "
        f"{len(confirmation_summary)} rows"
    )
    print(f"Saved: {summary_path}")

    # --------------------------------------------
    # Final merged result
    # --------------------------------------------

    final_df = merge_results(
        search_df,
        confirmation_summary,
    )

    final_path = (
        OUTPUT_DIR /
        "cifar10_final_summary.csv"
    )

    final_df.to_csv(
        final_path,
        index=False,
    )

    print(
        f"\nMerged final summary: "
        f"{len(final_df)} rows"
    )
    print(f"Saved: {final_path}")

    # --------------------------------------------
    # Paper-friendly
    # --------------------------------------------

    paper_df = make_paper_table(final_df)

    paper_path = (
        OUTPUT_DIR /
        "cifar10_paper_table.csv"
    )

    paper_df.to_csv(
        paper_path,
        index=False,
    )

    print(f"Saved: {paper_path}")

    # --------------------------------------------
    # Display
    # --------------------------------------------

    print("\n" + "=" * 70)
    print("CIFAR-10 FINAL SUMMARY")
    print("=" * 70)

    if not paper_df.empty:
        print(
            paper_df.to_string(
                index=False
            )
        )
    else:
        print("No results found.")


if __name__ == "__main__":
    main()