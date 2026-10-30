from pathlib import Path
import optuna

from optuna.trial import TrialState

from flbench.optimization.optuna_runner import (
    _storage,
    _trial_parameter_key,
    _load_yaml,
)


SEARCH_CONFIG = "configs/controlled_search_grid.yaml"

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

        algorithm_root = Path(
            "search_outputs"
        ) / scenario / algorithm

        if not algorithm_root.exists():
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

        waiting = [
            t for t in study.trials
            if t.state == TrialState.WAITING
        ]

        if not waiting:
            print(
                f"[OK] {study_name}: "
                "no WAITING trials"
            )
            continue

        seen = set()
        duplicates = []

        for trial in waiting:

            key = _trial_parameter_key(trial)

            if key in seen:
                duplicates.append(trial)
            else:
                seen.add(key)

        print(
            f"{study_name}: "
            f"WAITING={len(waiting)}, "
            f"unique={len(seen)}, "
            f"duplicates={len(duplicates)}"
        )

        # for trial in duplicates:

        #     study._storage.set_trial_user_attr(
        #         trial._trial_id,
        #         "run_status",
        #         "duplicate_retry_cancelled",
        #     )

        #     study._storage.set_trial_user_attr(
        #         trial._trial_id,
        #         "failure_type",
        #         "duplicate_retry",
        #     )

        #     study._storage.set_trial_user_attr(
        #         trial._trial_id,
        #         "failure_message",
        #         "Duplicate infrastructure retry removed.",
        #     )

        #     study._storage.set_trial_state_values(
        #         trial._trial_id,
        #         TrialState.FAIL,
        #     )

        # if duplicates:
        #     print(
        #         f"[CLEANED] {study_name}: "
        #         f"{len(duplicates)} duplicate retries removed."
        #     )