from pathlib import Path
import hashlib
import json
import numpy as np


def dirichlet_partition(
    targets,
    num_clients,
    alpha,
    seed,
    min_samples=1,
    attempts=1000,
):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    classes = np.unique(targets)

    for _ in range(attempts):
        out = {i: [] for i in range(num_clients)}
        for cls in classes:
            idx = np.where(targets == cls)[0]
            rng.shuffle(idx)
            counts = rng.multinomial(
                len(idx),
                rng.dirichlet(np.full(num_clients, alpha)),
            )
            start = 0
            for client_id, count in enumerate(counts):
                out[client_id].extend(
                    idx[start:start + count].tolist()
                )
                start += count

        if min(map(len, out.values())) >= min_samples:
            for values in out.values():
                rng.shuffle(values)
            return out

    raise RuntimeError(
        "Unable to satisfy minimum client size after Dirichlet "
        "partitioning; reduce min_samples_per_client, reduce "
        "num_clients, or increase alpha."
    )


def load_or_create(targets, cfg, dataset_name="dataset"):
    alpha = float(cfg["alpha"])
    clients = int(cfg["num_clients"])
    seed = int(cfg["seed"])
    namespace = str(
        cfg.get("cache_namespace", dataset_name)
    ).lower()

    key = (
        f"{namespace}|dirichlet|{alpha}|{clients}|"
        f"{seed}|{len(targets)}"
    )
    pid = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = Path(cfg["cache_dir"]) / namespace / f"{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            int(k): v
            for k, v in raw["client_indices"].items()
        }

    partition = dirichlet_partition(
        targets,
        clients,
        alpha,
        seed,
        int(cfg.get("min_samples_per_client", 1)),
    )

    path.write_text(
        json.dumps(
            {
                "partition_id": pid,
                "dataset": namespace,
                "method": "dirichlet",
                "alpha": alpha,
                "num_clients": clients,
                "seed": seed,
                "client_indices": partition,
                "client_sizes": {
                    str(k): len(v)
                    for k, v in partition.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return partition


def iid_partition(num_samples: int, num_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_clients)
    return {
        i: chunk.astype(int).tolist()
        for i, chunk in enumerate(chunks)
    }
