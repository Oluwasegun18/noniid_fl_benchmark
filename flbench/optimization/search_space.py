from __future__ import annotations

import itertools
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    kind: str
    values: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ParameterSpec":
        kind = str(raw.get("type", "categorical")).lower()
        if kind in {"categorical", "grid"}:
            values = tuple(raw.get("values", ()))
            if not values:
                raise ValueError(f"{kind} parameter requires non-empty values.")
            return cls(kind=kind, values=values)
        if kind in {"uniform", "log_uniform", "int_uniform"}:
            low, high = raw.get("low"), raw.get("high")
            if low is None or high is None or float(low) >= float(high):
                raise ValueError(f"Invalid bounds for {kind}: {low}, {high}")
            if kind == "log_uniform" and float(low) <= 0:
                raise ValueError("log_uniform requires low > 0.")
            return cls(kind=kind, low=float(low), high=float(high))
        raise ValueError(f"Unsupported parameter type: {kind}")

    def sample(self, rng: random.Random) -> Any:
        if self.kind in {"categorical", "grid"}:
            return rng.choice(self.values)
        if self.kind == "uniform":
            return rng.uniform(float(self.low), float(self.high))
        if self.kind == "log_uniform":
            return math.exp(rng.uniform(math.log(float(self.low)), math.log(float(self.high))))
        if self.kind == "int_uniform":
            return rng.randint(int(self.low), int(self.high))
        raise RuntimeError(self.kind)


def effective_search_space(search_cfg: dict[str, Any], algorithm: str) -> dict[str, ParameterSpec]:
    raw: dict[str, Any] = {}
    raw.update(search_cfg.get("shared_search_space", {}))
    raw.update(search_cfg.get("algorithm_search_spaces", {}).get(algorithm, {}))
    return {key: ParameterSpec.from_dict(value) for key, value in raw.items()}


def grid_candidates(search_cfg: dict[str, Any], algorithm: str) -> list[dict[str, Any]]:
    space = effective_search_space(search_cfg, algorithm)
    if not space:
        return [{}]

    invalid = [
        key for key, spec in space.items()
        if spec.kind not in {"categorical", "grid"}
    ]
    if invalid:
        raise ValueError(
            "Exhaustive grid search requires explicit `values` for every "
            f"parameter. Continuous specifications found for: {invalid}"
        )

    keys = sorted(space)
    value_lists = [space[key].values for key in keys]
    return [
        dict(zip(keys, values))
        for values in itertools.product(*value_lists)
    ]


def sample_candidates(
    search_cfg: dict[str, Any],
    algorithm: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    space = effective_search_space(search_cfg, algorithm)
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(100, count * 50)

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        sampled = {
            key: spec.sample(rng)
            for key, spec in sorted(space.items())
        }
        signature = repr(sorted(sampled.items()))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(sampled)

    if len(candidates) < count:
        raise ValueError(
            f"Could only generate {len(candidates)} unique candidates for "
            f"{algorithm}; requested {count}."
        )
    return candidates


def generate_candidates(
    search_cfg: dict[str, Any],
    algorithm: str,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    method = str(
        search_cfg.get("search", {}).get(
            "method",
            search_cfg.get("budget", {}).get("search_method", "grid"),
        )
    ).lower()

    if method == "grid":
        return grid_candidates(search_cfg, algorithm)

    if method == "random":
        count = search_cfg.get("search", {}).get("max_configurations")
        if count is None:
            count = search_cfg.get("budget", {}).get(
                "max_configurations_per_algorithm"
            )
        if count is None:
            raise ValueError(
                "Random search requires a maximum candidate count."
            )
        return sample_candidates(search_cfg, algorithm, int(count), seed)

    raise ValueError(f"Unsupported search method: {method}")


def apply_parameters(
    config: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    output = deepcopy(config)
    for dotted_key, value in parameters.items():
        current = output
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return output
