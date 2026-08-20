from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


REQUIRED = {
    "experiment", "data", "partition", "model", "training",
    "federation", "algorithm", "stopping", "evaluation", "output",
}
SUPPORTED_DATASETS = {"femnist", "cifar10", "cifar100", "covtype", "shakespeare"}
SUPPORTED_ALGORITHMS = {
    "fedavg", "fedprox", "scaffold", "fednova",
    "feddyn", "moon", "fedsam", "fedgucci",
}
SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw"}


def load_config(path):
    """Load YAML and validate the benchmark configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Validate structural and convergence-related configuration values."""
    missing = REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    dataset = str(cfg["data"]["dataset"]).lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    algorithm = str(cfg["algorithm"]["name"]).lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    optimizer = str(cfg["training"].get("optimizer", "sgd")).lower()
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    participation = float(cfg["federation"]["participation_rate"])
    if not 0 < participation <= 1:
        raise ValueError("participation_rate must be in (0,1].")

    if int(cfg["partition"].get("num_clients", 1)) < 1:
        raise ValueError("num_clients must be positive.")

    if int(cfg["evaluation"].get("frequency", 1)) < 1:
        raise ValueError("evaluation.frequency must be >= 1.")

    stopping = cfg["stopping"]
    if str(stopping.get("mode", "fixed_rounds")).lower() == "convergence":
        min_delta = float(
            stopping.get(
                "min_delta",
                stopping.get("convergence_tolerance", 1e-4),
            )
        )
        if min_delta < 0:
            raise ValueError("stopping.min_delta must be >= 0.")

        patience = int(
            stopping.get(
                "patience_evaluations",
                stopping.get("patience", 1),
            )
        )
        if patience < 1:
            raise ValueError("stopping.patience_evaluations must be >= 1.")

        if int(stopping.get("min_rounds", 0)) < 0:
            raise ValueError("stopping.min_rounds must be >= 0.")


def deep_set(cfg, dotted, value):
    """Return a validated copy with one dotted configuration path updated."""
    out = deepcopy(cfg)
    current = out
    keys = dotted.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value
    validate_config(out)
    return out
