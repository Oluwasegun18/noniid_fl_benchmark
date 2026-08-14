from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass
class TrainResult:
    state_dict: dict[str,torch.Tensor]
    metrics: dict[str,float]
    auxiliary: dict[str,Any]=field(default_factory=dict)

@dataclass
class ClientUpdate:
    client_id:int
    num_samples:int
    state_dict:dict[str,torch.Tensor]
    metrics:dict[str,float]
    auxiliary:dict[str,Any]=field(default_factory=dict)
