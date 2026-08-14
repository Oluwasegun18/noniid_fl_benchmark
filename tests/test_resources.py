import torch

from flbench.experiment.resources import (
    EnergyTracker,
    communication_metrics,
    payload_num_bytes,
)


def minimal_config():
    return {
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
                    "sample_interval_sec": 0.05,
                },
            },
            "communication": {
                "downlink_mbps": 100.0,
                "uplink_mbps": 20.0,
                "one_way_latency_ms": 10.0,
                "downlink_j_per_byte": 1e-7,
                "uplink_j_per_byte": 2e-7,
            },
        }
    }


def test_payload_num_bytes():
    payload = {
        "a": torch.zeros(4, dtype=torch.float32),
        "b": [torch.zeros(2, dtype=torch.float64)],
    }
    assert payload_num_bytes(payload) == 4 * 4 + 2 * 8


def test_modeled_energy_fallback():
    tracker = EnergyTracker(
        minimal_config(),
        torch.device("cpu"),
        phase="test",
    )
    tracker.start()

    # Small deterministic CPU workload.
    value = 0
    for index in range(200_000):
        value += index * index
    assert value > 0

    result = tracker.stop()
    assert result.compute_energy_reported_j is not None
    assert result.compute_energy_reported_j >= 0
    assert result.energy_is_estimated is True
    assert "cpu_modeled" in result.energy_source


def test_communication_metrics():
    result = communication_metrics(
        download_bytes=1_000_000,
        upload_bytes=500_000,
        selected_clients=5,
        cfg=minimal_config(),
    )
    assert result["total_comm_bytes"] == 1_500_000
    assert result["modeled_comm_time_sequential_sec"] > 0
    assert result["modeled_comm_time_parallel_sec"] > 0
    assert result["modeled_comm_energy_j"] > 0
