from flbench.optimization.search_space import (
    apply_parameters,
    effective_search_space,
    sample_candidates,
)
from flbench.optimization.selector import rank_candidates


def search_config():
    return {
        "shared_search_space": {
            "training.learning_rate": {
                "type": "categorical",
                "values": [0.01, 0.1],
            }
        },
        "algorithm_search_spaces": {
            "fedprox": {
                "algorithm.parameters.fedprox_mu": {
                    "type": "categorical",
                    "values": [0.001, 0.01],
                }
            }
        },
    }


def test_effective_space_combines_shared_and_specific():
    space = effective_search_space(search_config(), "fedprox")
    assert set(space) == {
        "training.learning_rate",
        "algorithm.parameters.fedprox_mu",
    }


def test_candidate_sampling_is_reproducible():
    first = sample_candidates(search_config(), "fedprox", 3, seed=7)
    second = sample_candidates(search_config(), "fedprox", 3, seed=7)
    assert first == second


def test_apply_parameters_does_not_mutate_base():
    base = {"training": {"learning_rate": 1.0}}
    changed = apply_parameters(base, {"training.learning_rate": 0.1})
    assert changed["training"]["learning_rate"] == 0.1
    assert base["training"]["learning_rate"] == 1.0


def test_selector_uses_validation_not_test():
    records = [
        {"candidate_id": "a", "status": "completed", "val_accuracy": 0.8, "test_accuracy": 0.99},
        {"candidate_id": "b", "status": "completed", "val_accuracy": 0.9, "test_accuracy": 0.10},
    ]
    ranked = rank_candidates(records, {"metric": "val_accuracy", "mode": "maximize"})
    assert ranked[0]["candidate_id"] == "b"
