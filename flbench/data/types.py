from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from torch.utils.data import Dataset

@dataclass
class FederatedDatasetBundle:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    client_indices: dict[int, list[int]]
    task_type: str
    num_classes: int
    input_shape: tuple[int, ...] | None = None
    vocabulary: list[str] | None = None
    client_names: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
