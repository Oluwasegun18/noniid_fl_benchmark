from __future__ import annotations

from typing import Any


def _value(record: dict[str, Any], key: str, default: float) -> float:
    value = record.get(key)
    return default if value is None else float(value)


def rank_candidates(records: list[dict[str, Any]], selection_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    completed = [record for record in records if record.get("status") == "completed"]
    metric = selection_cfg.get("metric", "val_accuracy")
    mode = selection_cfg.get("mode", "maximize")
    tie_breakers = selection_cfg.get(
        "tie_breakers",
        ["val_loss", "cumulative_wall_time_sec", "total_comm_bytes"],
    )

    constraints = selection_cfg.get("constraints", {})
    feasible = []
    for record in completed:
        valid = True
        for field, maximum in constraints.items():
            if maximum is not None and record.get(field) is not None:
                valid &= float(record[field]) <= float(maximum)
        if valid:
            feasible.append(record)

    def key(record: dict[str, Any]):
        primary = _value(record, metric, float("-inf") if mode == "maximize" else float("inf"))
        primary_key = -primary if mode == "maximize" else primary
        return tuple([primary_key] + [_value(record, field, float("inf")) for field in tie_breakers])

    ranked = sorted(feasible, key=key)
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank
    return ranked
