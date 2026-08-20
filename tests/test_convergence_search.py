from flbench.experiment.stopping import StoppingController
from flbench.optimization.search_space import grid_candidates


def _cfg():
    return {
        "stopping": {
            "mode": "convergence",
            "min_delta": 0.01,
            "patience_evaluations": 3,
            "min_rounds": 5,
        }
    }


def test_convergence_patience_counts_only_observed_metrics():
    controller = StoppingController(_cfg())
    assert controller.update(1, None, 0.0, 100) == (False, None)
    assert controller.update(2, 0.50, 0.0, 100) == (False, None)
    assert controller.best_round == 2
    # Unevaluated rounds must not consume patience.
    assert controller.update(3, None, 0.0, 100) == (False, None)
    assert controller.wait == 0
    controller.update(4, 0.505, 0.0, 100)
    controller.update(5, 0.506, 0.0, 100)
    stopped, reason = controller.update(6, 0.507, 0.0, 100)
    assert stopped is True
    assert reason == "convergence"


def test_max_rounds_is_safety_ceiling():
    controller = StoppingController(_cfg())
    stopped, reason = controller.update(100, 0.9, 1.0, 100)
    assert stopped is True
    assert reason == "max_rounds"
    assert controller.best_round == 100


def test_primary_grid_sizes():
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        Path("configs/controlled_search_grid.yaml").read_text(encoding="utf-8")
    )
    expected = {
        "fedavg": 36,
        "fedprox": 144,
        "scaffold": 36,
        "fednova": 36,
        "feddyn": 108,
        "moon": 144,
        "fedsam": 108,
        "fedgucci": 576,
    }
    assert {
        algorithm: len(grid_candidates(cfg, algorithm))
        for algorithm in cfg["algorithms"]
    } == expected
