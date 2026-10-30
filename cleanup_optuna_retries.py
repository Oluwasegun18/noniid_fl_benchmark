from pathlib import Path

import optuna
from optuna.trial import TrialState

from flbench.optimization.optuna_runner import (
    _storage,
    _trial_parameter_key,
    _load_yaml,
)


SEARCH_CONFIG = "configs/controlled_search_grid.yaml"

# ---------------------------------------------------------
# SAFETY:
# True  -> only report what WOULD be changed
# False -> actually cancel redundant WAITING trials
# ---------------------------------------------------------
DRY_RUN = True


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


SCENARIOS = [
    "cifar100_dirichlet_highly_non_iid_a0p01",
    "cifar100_dirichlet_mildly_non_iid_a0p5",
    "cifar100_dirichlet_iid_like_a100p0",
]


search_cfg = _load_yaml(SEARCH_CONFIG)


for scenario in SCENARIOS:

    for algorithm in ALGORITHMS:

        algorithm_root = (
            Path("search_outputs")
            / scenario
            / algorithm
        )

        if not algorithm_root.exists():
            print(
                f"[SKIP] {scenario}__{algorithm}: "
                "algorithm directory does not exist"
            )
            continue

        study_name = f"{scenario}__{algorithm}"

        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=_storage(
                    search_cfg,
                    algorithm_root.resolve(),
                ),
            )

        except Exception as exc:
            print(
                f"[SKIP] {study_name}: "
                f"could not load study: {exc}"
            )
            continue

        # =====================================================
        # 1. Current WAITING trials
        # =====================================================

        waiting = [
            t
            for t in study.trials
            if t.state == TrialState.WAITING
        ]

        if not waiting:
            print(
                f"[OK] {study_name}: "
                "no WAITING trials"
            )
            continue

        # =====================================================
        # 2. Configurations that have already completed
        # =====================================================

        completed_keys = {
            _trial_parameter_key(t)
            for t in study.trials
            if t.state == TrialState.COMPLETE
        }

        # =====================================================
        # 3. Group WAITING trials by hyperparameter configuration
        # =====================================================

        by_key = {}

        for trial in waiting:

            key = _trial_parameter_key(trial)

            by_key.setdefault(
                key,
                []
            ).append(trial)

        # =====================================================
        # 4. Decide which WAITING trials to keep/cancel
        # =====================================================

        keep_trials = []
        cancel_trials = []

        already_completed_trials = []
        duplicate_waiting_trials = []

        for key, trials in by_key.items():

            # -------------------------------------------------
            # CASE A:
            # The configuration has already completed.
            #
            # There is no reason to execute ANY waiting copy.
            # -------------------------------------------------

            if key in completed_keys:

                cancel_trials.extend(trials)
                already_completed_trials.extend(trials)

                continue

            # -------------------------------------------------
            # CASE B:
            # The configuration still needs evaluation.
            #
            # Keep exactly ONE WAITING trial.
            # -------------------------------------------------

            tagged = [
                t
                for t in trials
                if t.user_attrs.get("retry_reason")
                == "infrastructure_interruption"
            ]

            # Prefer a retry that explicitly records
            # the infrastructure-interruption history.
            if tagged:
                keep = tagged[0]
            else:
                keep = trials[0]

            keep_trials.append(keep)

            # Cancel every other waiting copy.
            for trial in trials:

                if trial.number != keep.number:

                    cancel_trials.append(trial)
                    duplicate_waiting_trials.append(trial)

        # =====================================================
        # 5. Report
        # =====================================================

        print()
        print("=" * 70)
        print(study_name)
        print("=" * 70)

        print(
            f"WAITING before cleanup       : {len(waiting)}"
        )

        print(
            f"Unique WAITING configurations: {len(by_key)}"
        )

        print(
            f"WAITING configurations to keep: {len(keep_trials)}"
        )

        print(
            f"Duplicate WAITING to cancel   : "
            f"{len(duplicate_waiting_trials)}"
        )

        print(
            f"Already-completed WAITING     : "
            f"{len(already_completed_trials)}"
        )

        print(
            f"Total WAITING to cancel       : "
            f"{len(cancel_trials)}"
        )

        # =====================================================
        # 6. Show exactly what will be cancelled
        # =====================================================

        if cancel_trials:

            print("\nTrials selected for cleanup:")

            for trial in cancel_trials:

                key = _trial_parameter_key(trial)

                if trial in already_completed_trials:
                    reason = "configuration_already_completed"
                else:
                    reason = "duplicate_waiting_retry"

                print(
                    f"  trial={trial.number:<6} "
                    f"reason={reason:<32} "
                    f"params={key}"
                )

        # =====================================================
        # 7. DRY RUN
        # =====================================================

        if DRY_RUN:

            print(
                "\n[DRY RUN] No Optuna trial states were changed."
            )

            continue

        # =====================================================
        # 8. Actually cancel redundant WAITING trials
        # =====================================================

        for trial in cancel_trials:

            if trial in already_completed_trials:

                message = (
                    "Queued retry cancelled because the same "
                    "hyperparameter configuration already has "
                    "a successful COMPLETE trial."
                )

            else:

                message = (
                    "Duplicate queued infrastructure retry "
                    "removed during recovery cleanup."
                )

            # WAITING trials are still mutable.
            study._storage.set_trial_user_attr(
                trial._trial_id,
                "run_status",
                "duplicate_retry_cancelled",
            )

            study._storage.set_trial_user_attr(
                trial._trial_id,
                "failure_type",
                "duplicate_retry",
            )

            study._storage.set_trial_user_attr(
                trial._trial_id,
                "failure_message",
                message,
            )

            study._storage.set_trial_state_values(
                trial._trial_id,
                TrialState.FAIL,
            )

        # =====================================================
        # 9. Verify resulting queue
        # =====================================================

        waiting_after = [
            t
            for t in study.trials
            if t.state == TrialState.WAITING
        ]

        print(
            f"\n[CLEANED] {study_name}"
        )

        print(
            f"WAITING before: {len(waiting)}"
        )

        print(
            f"Cancelled     : {len(cancel_trials)}"
        )

        print(
            f"WAITING after : {len(waiting_after)}"
        )