from pathlib import Path
import math

import numpy as np
from torchvision import datasets

from flbench.data.partitioning import dirichlet_partition


DATA_DIR = "./data"
NUM_CLIENTS = 10
PARTITION_SEED = 1
SPLIT_SEED = 42
VAL_FRACTION = 0.1


def split_indices(length, validation_fraction, seed):
    rng = np.random.default_rng(seed)

    idx = np.arange(length)
    rng.shuffle(idx)

    nval = max(
        1,
        int(len(idx) * validation_fraction)
    )

    return idx[nval:], idx[:nval]


def normalized_entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    p = p[p > 0]

    return (
        -np.sum(p * np.log2(p))
        / math.log2(10)
    )


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(
            a[mask]
            * np.log2(a[mask] / b[mask])
        )

    return (
        0.5 * kl(p, m)
        + 0.5 * kl(q, m)
    )


dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
)

targets = np.asarray(dataset.targets)

train_idx, _ = split_indices(
    len(targets),
    VAL_FRACTION,
    SPLIT_SEED,
)

train_targets = targets[train_idx]

global_counts = np.bincount(
    train_targets,
    minlength=10,
)

global_dist = global_counts / global_counts.sum()


for alpha in [0.01, 0.5, 100.0]:

    # IMPORTANT:
    # generate fresh directly -- do NOT use cache
    partition = dirichlet_partition(
        targets=train_targets,
        num_clients=NUM_CLIENTS,
        alpha=alpha,
        seed=PARTITION_SEED,
        min_samples=1,
    )

    print("\n" + "=" * 70)
    print(f"ALPHA = {alpha}")
    print("=" * 70)

    entropies = []
    js_values = []
    nclasses = []
    sizes = []

    for cid in sorted(partition):

        indices = np.asarray(
            partition[cid],
            dtype=int,
        )

        labels = train_targets[indices]

        counts = np.bincount(
            labels,
            minlength=10,
        )

        probs = counts / counts.sum()

        nc = np.sum(counts > 0)
        ent = normalized_entropy(counts)
        js = js_divergence(
            probs,
            global_dist,
        )

        nclasses.append(nc)
        entropies.append(ent)
        js_values.append(js)
        sizes.append(len(indices))

        print(
            f"client {cid:2d} | "
            f"n={len(indices):5d} | "
            f"classes={nc:2d} | "
            f"entropy={ent:.4f} | "
            f"JS={js:.4f} | "
            f"counts={counts.tolist()}"
        )

    print("\nSUMMARY")

    print(
        "mean classes/client:",
        round(float(np.mean(nclasses)), 4)
    )

    print(
        "mean entropy:",
        round(float(np.mean(entropies)), 4)
    )

    print(
        "mean JS:",
        round(float(np.mean(js_values)), 4)
    )

    print(
        "client sizes:",
        min(sizes),
        "-",
        max(sizes)
    )