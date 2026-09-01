import torch

from flbench.algorithms.base import FederatedAlgorithm
from flbench.optimization.optuna_runner import _grid_space
from flbench.training.types import ClientUpdate
from flbench.experiment.resources import EnergyTracker


def test_device_safe_weighted_average_cpu():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    updates = [
        ClientUpdate(client_id=0, state_dict={"weight": torch.tensor([[1.0, 3.0]])}, num_samples=1, metrics={}),
        ClientUpdate(client_id=1, state_dict={"weight": torch.tensor([[3.0, 5.0]])}, num_samples=3, metrics={}),
    ]
    FederatedAlgorithm.weighted_average(model, updates)
    expected = torch.tensor([[2.5, 4.5]])
    assert torch.allclose(model.weight.detach(), expected)


def test_optuna_grid_space_preserves_predefined_grid():
    cfg = {
        "shared_search_space": {
            "training.learning_rate": {"type": "grid", "values": [0.01, 0.04, 0.1]},
        },
        "algorithm_search_spaces": {
            "fedprox": {
                "algorithm.parameters.fedprox_mu": {"type": "grid", "values": [0.001, 0.01]},
            }
        },
    }
    space = _grid_space(cfg, "fedprox")
    assert space["training.learning_rate"] == [0.01, 0.04, 0.1]
    assert space["algorithm.parameters.fedprox_mu"] == [0.001, 0.01]


def test_modeled_energy_is_not_ranking_valid_without_gpu():
    cfg = {
        "resources": {
            "energy": {
                "role": "ranking",
                "mode": "modeled_only",
                "allow_modeled_fallback": True,
                "cpu": {"power_model": "constant", "active_power_watts": 5.0},
            }
        }
    }
    tracker = EnergyTracker(cfg, torch.device("cpu"), phase="confirmation")
    tracker.start()
    _ = sum(i * i for i in range(10000))
    reading = tracker.stop()
    assert reading.gpu_energy_primary_j is None
    assert reading.energy_valid_for_ranking is False
    assert "gpu_energy_unavailable" in (reading.energy_invalid_reason or "")
