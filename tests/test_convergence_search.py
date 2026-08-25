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


def test_smoothed_convergence_ignores_small_noniid_oscillations():
    cfg = {
        "stopping": {
            "mode": "convergence",
            "smoothing_window": 3,
            "min_delta": 0.001,
            "patience_evaluations": 4,
            "min_rounds": 1,
        }
    }
    controller = StoppingController(cfg)

    # A noisy plateau around 0.73.  Individual observations rise and fall,
    # but the moving average does not improve by 0.1 percentage point.
    values = [0.7300, 0.7310, 0.7290, 0.7305, 0.7295, 0.7308, 0.7297]
    result = (False, None)
    for idx, value in enumerate(values, start=1):
        result = controller.update(idx, value, 0.0, 100)
        if result[0]:
            break

    assert result == (True, "convergence")
    assert controller.wait >= 4


def test_raw_best_is_independent_of_smoothed_stopping_best():
    cfg = {
        "stopping": {
            "mode": "convergence",
            "smoothing_window": 3,
            "min_delta": 0.01,
            "patience_evaluations": 10,
            "min_rounds": 1,
        }
    }
    controller = StoppingController(cfg)
    controller.update(1, 0.70, 0.0, 100)
    controller.update(2, 0.72, 0.0, 100)
    controller.update(3, 0.71, 0.0, 100)

    # Raw-best checkpoint should correspond to the single strongest observed
    # validation value, not to the moving-average stopping statistic.
    assert controller.raw_best == 0.72
    assert controller.raw_best_round == 2
    assert controller.best != controller.raw_best


def test_smoothing_state_survives_resume():
    cfg = {
        "stopping": {
            "mode": "convergence",
            "smoothing_window": 3,
            "min_delta": 0.001,
            "patience_evaluations": 10,
            "min_rounds": 1,
        }
    }
    original = StoppingController(cfg)
    original.update(1, 0.70, 0.0, 100)
    original.update(2, 0.71, 0.0, 100)
    original.update(3, 0.705, 0.0, 100)

    restored = StoppingController(cfg)
    restored.load_state_dict(original.state_dict())

    assert list(restored.metric_history) == list(original.metric_history)
    assert restored.raw_best == original.raw_best
    assert restored.best == original.best
    assert restored.wait == original.wait
