from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SearchBudget:
    max_configurations: int
    max_total_runs: int
    max_wall_time_hours: float | None = None
    started_at: float = field(default_factory=time.perf_counter)
    attempted_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def can_start(self) -> bool:
        if self.attempted_runs >= self.max_total_runs:
            return False
        if self.max_wall_time_hours is not None:
            return self.elapsed_seconds < self.max_wall_time_hours * 3600.0
        return True

    def register(self, status: str) -> None:
        self.attempted_runs += 1
        if status == "completed":
            self.completed_runs += 1
        else:
            self.failed_runs += 1

    def to_dict(self) -> dict:
        return {
            "max_configurations": self.max_configurations,
            "max_total_runs": self.max_total_runs,
            "max_wall_time_hours": self.max_wall_time_hours,
            "attempted_runs": self.attempted_runs,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "elapsed_seconds": self.elapsed_seconds,
        }
