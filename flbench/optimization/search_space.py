from __future__ import annotations

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
        if kind == "categorical":
            values = tuple(raw.get("values", ()))
            if not values:
                raise ValueError("Categorical parameter requires non-empty values.")
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
        if self.kind == "categorical":
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
        sampled = {key: spec.sample(rng) for key, spec in sorted(space.items())}
        signature = repr(sorted(sampled.items()))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(sampled)

    if len(candidates) < count:
        raise ValueError(
            f"Could only generate {len(candidates)} unique candidates for {algorithm}; "
            f"requested {count}. Reduce the budget or enlarge the search space."
        )
    return candidates


def apply_parameters(config: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(config)
    for dotted_key, value in parameters.items():
        current = output
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return output
