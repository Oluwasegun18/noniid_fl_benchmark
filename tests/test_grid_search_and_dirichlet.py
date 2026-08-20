import numpy as np

from flbench.data.partitioning import dirichlet_partition
from flbench.optimization.search_space import grid_candidates


def test_grid_size_can_differ_by_algorithm():
    cfg = {
        "search": {"method": "grid"},
        "shared_search_space": {
            "training.learning_rate": {
                "type": "grid",
                "values": [0.01, 0.1],
            },
            "training.local_epochs": {
                "type": "grid",
                "values": [1, 2],
            },
        },
        "algorithm_search_spaces": {
            "fedavg": {},
            "fedprox": {
                "algorithm.parameters.fedprox_mu": {
                    "type": "grid",
                    "values": [0.01, 0.1, 1.0],
                }
            },
        },
    }
    assert len(grid_candidates(cfg, "fedavg")) == 4
    assert len(grid_candidates(cfg, "fedprox")) == 12


def test_three_dirichlet_severity_alphas():
    targets = np.tile(np.arange(10), 200)
    for alpha in (0.1, 0.5, 100.0):
        partition = dirichlet_partition(
            targets,
            num_clients=10,
            alpha=alpha,
            seed=7,
            min_samples=5,
        )
        assert len(partition) == 10
        assert (
            sum(len(v) for v in partition.values())
            == len(targets)
        )
