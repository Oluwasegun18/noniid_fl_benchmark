from pathlib import Path

from flbench.optimization.search_runner import _prepare_run_config


def base_config():
    return {
        "experiment": {
            "name": "base",
            "trial": 1,
            "seed": 1,
            "output_dir": "outputs",
            "device": "cpu",
            "deterministic": True,
        },
        "data": {
            "dataset": "cifar10",
            "data_dir": "./data",
            "validation_fraction": 0.1,
            "preprocessing": "standard",
            "augmentation": False,
            "subset_size": 100,
        },
        "partition": {
            "method": "dirichlet",
            "heterogeneity_class": "label_skew",
            "alpha": 0.5,
            "num_clients": 5,
            "min_samples_per_client": 1,
            "seed": 1,
            "cache_dir": "./partition_cache",
        },
        "model": {
            "architecture": "simple_cnn",
            "initialization_seed": 1,
            "num_classes": 10,
        },
        "training": {
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "local_epochs": 1,
            "batch_size": 32,
            "num_workers": 0,
        },
        "federation": {
            "participation_rate": 1.0,
            "communication_rounds": 2,
            "client_sampling_seed": 1,
            "synchronization": "synchronous",
        },
        "algorithm": {
            "name": "fedavg",
            "parameters": {},
        },
        "stopping": {
            "mode": "fixed_rounds",
            "target_accuracy": None,
            "convergence_tolerance": 0.001,
            "patience": 2,
            "runtime_budget_sec": None,
        },
        "evaluation": {
            "frequency": 1,
            "validation_metric": "accuracy",
            "reporting_rule": "final_round",
            "num_trials": 1,
        },
        "output": {
            "save_checkpoints": False,
            "checkpoint_frequency": 1,
            "resume": False,
        },
        "resources": {
            "energy": {
                "mode": "modeled_only",
                "allow_modeled_fallback": True,
                "cpu": {
                    "power_model": "constant",
                    "active_power_watts": 10.0,
                    "idle_power_watts": 2.0,
                    "include_idle_power": False,
                },
                "gpu": {
                    "device_index": 0,
                    "sample_interval_sec": 0.1,
                },
            },
            "communication": {
                "downlink_mbps": 100.0,
                "uplink_mbps": 20.0,
                "one_way_latency_ms": 10.0,
                "downlink_j_per_byte": 0.0,
                "uplink_j_per_byte": 0.0,
            },
        },
    }


def test_search_uses_exact_short_output_directory(tmp_path):
    run_dir = tmp_path / "candidate_runs" / "c0001" / "s1"
    cfg = _prepare_run_config(
        base_config(),
        "fedavg",
        "c0001_s1",
        {"training.batch_size": 64},
        seed=1,
        rounds=3,
        output_dir=run_dir,
    )

    assert cfg["experiment"]["use_exact_output_dir"] is True
    assert Path(cfg["experiment"]["output_dir"]) == run_dir
    assert cfg["training"]["batch_size"] == 64
    assert cfg["federation"]["communication_rounds"] == 3
