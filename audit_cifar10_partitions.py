from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
from torchvision import datasets


SCENARIOS = {
    "cifar10_dirichlet_highly_non_iid_a0p01": "alpha=0.01",
    "cifar10_dirichlet_mildly_non_iid_a0p5": "alpha=0.5",
    "cifar10_dirichlet_iid_like_a100p0": "alpha=100",
}


def split_indices(length, validation_fraction, seed, subset_size=None):
    rng = np.random.default_rng(seed)

    idx = np.arange(length)
    rng.shuffle(idx)

    if subset_size is not None:
        idx = idx[:min(int(subset_size), len(idx))]

    nval = max(
        1,
        int(len(idx) * validation_fraction)
    )

    train_idx = idx[nval:]
    val_idx = idx[:nval]

    return train_idx, val_idx


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(
            a[mask] *
            np.log2(a[mask] / b[mask])
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalized_entropy(p):
    p = np.asarray(p, dtype=float)

    if p.sum() == 0:
        return np.nan

    p = p / p.sum()
    p = p[p > 0]

    h = -np.sum(p * np.log2(p))

    return h / math.log2(10)


rows = []
client_rows = []

for scenario, label in SCENARIOS.items():

    # Use FedAvg only: partition should be shared across algorithms.
    # root = (
    #     Path("search_outputs")
    #     / scenario
    #     / "fedavg"
    #     / "confirmation_runs"
    # )

    root = (
            Path("search_outputs")
            / scenario
            / "fedavg"
            / "candidate_runs"
            / "trial_00000"
            
        )

    # for run_dir in sorted(root.glob("t*_s*")):

    run_dirs = [root / "s42"]

    for run_dir in run_dirs:
        if not run_dir.exists():
            print("Missing:", run_dir)
            continue

        cfg_path = run_dir / "resolved_config.json"

        if not cfg_path.exists():
            continue

        cfg = json.loads(
            cfg_path.read_text()
        )

        experiment_seed = int(
            cfg["experiment"]["seed"]
        )

        part = cfg["partition"]

        alpha = float(part["alpha"])
        num_clients = int(part["num_clients"])
        partition_seed = int(part["seed"])

        data_cfg = cfg["data"]

        data_dir = str(
            data_cfg["data_dir"]
        )

        validation_fraction = float(
            data_cfg.get(
                "validation_fraction",
                0.1,
            )
        )

        subset_size = data_cfg.get(
            "subset_size"
        )

        # --------------------------------------------------
        # Load original CIFAR-10 labels
        # --------------------------------------------------

        base = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=False,
        )

        original_targets = np.asarray(
            base.targets,
            dtype=int,
        )

        # --------------------------------------------------
        # Reproduce the exact train/validation split
        # --------------------------------------------------

        train_idx, val_idx = split_indices(
            len(original_targets),
            validation_fraction,
            experiment_seed,
            subset_size,
        )

        train_targets = original_targets[
            train_idx
        ]

        # --------------------------------------------------
        # Locate exact cached partition
        # --------------------------------------------------

        namespace = str(
            part.get(
                "cache_namespace",
                "cifar10",
            )
        ).lower()

        key = (
            f"{namespace}|dirichlet|"
            f"{alpha}|{num_clients}|"
            f"{partition_seed}|"
            f"{len(train_targets)}"
        )

        pid = hashlib.sha1(
            key.encode()
        ).hexdigest()[:16]

        cache_file = (
            Path(part["cache_dir"])
            / namespace
            / f"{pid}.json"
        )

        if not cache_file.exists():
            print(
                "MISSING CACHE:",
                cache_file,
            )
            continue

        cached = json.loads(
            cache_file.read_text()
        )

        client_indices = {
            int(k): np.asarray(v, dtype=int)
            for k, v in cached[
                "client_indices"
            ].items()
        }

        # Global distribution for this actual training split
        global_counts = np.bincount(
            train_targets,
            minlength=10,
        )

        global_dist = (
            global_counts
            / global_counts.sum()
        )

        entropies = []
        js_values = []
        classes_per_client = []
        client_sizes = []

        # --------------------------------------------------
        # Per-client distributions
        # --------------------------------------------------

        for client_id, indices in sorted(
            client_indices.items()
        ):

            labels = train_targets[indices]

            counts = np.bincount(
                labels,
                minlength=10,
            )

            probs = counts / counts.sum()

            n_classes = int(
                np.sum(counts > 0)
            )

            entropy = normalized_entropy(
                counts
            )

            js = js_divergence(
                probs,
                global_dist,
            )

            entropies.append(entropy)
            js_values.append(js)
            classes_per_client.append(
                n_classes
            )
            client_sizes.append(
                len(labels)
            )

            row = {
                "scenario": label,
                "run": run_dir.name,
                "experiment_seed":
                    experiment_seed,
                "partition_seed":
                    partition_seed,
                "alpha": alpha,
                "client":
                    client_id,
                "num_samples":
                    len(labels),
                "num_classes":
                    n_classes,
                "normalized_entropy":
                    entropy,
                "js_divergence":
                    js,
            }

            for c in range(10):
                row[f"class_{c}"] = int(
                    counts[c]
                )

            client_rows.append(row)

        rows.append({
            "scenario": label,
            "run": run_dir.name,
            "experiment_seed":
                experiment_seed,
            "partition_seed":
                partition_seed,
            "alpha":
                alpha,

            "num_clients":
                num_clients,

            "mean_samples_per_client":
                np.mean(client_sizes),

            "min_samples_per_client":
                np.min(client_sizes),

            "max_samples_per_client":
                np.max(client_sizes),

            "mean_classes_per_client":
                np.mean(
                    classes_per_client
                ),

            "min_classes_per_client":
                np.min(
                    classes_per_client
                ),

            "max_classes_per_client":
                np.max(
                    classes_per_client
                ),

            "mean_normalized_entropy":
                np.mean(entropies),

            "mean_js_divergence":
                np.mean(js_values),

            "max_js_divergence":
                np.max(js_values),

            "partition_id":
                pid,
        })


summary = pd.DataFrame(rows)
clients = pd.DataFrame(client_rows)

summary.to_csv(
    "cifar10_partition_audit.csv",
    index=False,
)

clients.to_csv(
    "cifar10_client_label_distributions.csv",
    index=False,
)

print("\n================================================")
print("PARTITION AUDIT")
print("================================================")

cols = [
    "scenario",
    "run",
    "experiment_seed",
    "partition_seed",
    "mean_classes_per_client",
    "mean_normalized_entropy",
    "mean_js_divergence",
    "min_samples_per_client",
    "max_samples_per_client",
]

print(
    summary[cols]
    .round(4)
    .to_string(index=False)
)

print("\n================================================")
print("AVERAGE BY HETEROGENEITY REGIME")
print("================================================")

print(
    summary.groupby("scenario")
    .agg(
        classes_per_client=(
            "mean_classes_per_client",
            "mean",
        ),
        entropy=(
            "mean_normalized_entropy",
            "mean",
        ),
        js_divergence=(
            "mean_js_divergence",
            "mean",
        ),
        min_client_size=(
            "min_samples_per_client",
            "mean",
        ),
        max_client_size=(
            "max_samples_per_client",
            "mean",
        ),
    )
    .round(4)
    .to_string()
)