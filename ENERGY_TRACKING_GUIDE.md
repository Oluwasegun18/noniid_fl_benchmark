# Energy and Time Tracking Guide

## Windows CPU testing

Use `configs/windows_cpu_smoke.yaml`. Energy is estimated using:

`estimated_energy = estimated_power * elapsed_time`

With `power_model: process_utilization`, estimated power is proportional to
the Python process CPU-time share across logical cores. This is appropriate
for testing the logging pipeline, but it is not a hardware energy measurement.

## Linux GPU cluster

The tracker records GPU energy using NVML. If the device supports cumulative
energy, it uses the counter difference. Otherwise, it samples GPU power and
integrates it over time. CPU energy uses Linux RAPL-compatible package
counters when accessible.

On a shared node or shared parent GPU, device-level counters can include
other workloads. Use exclusive GPU allocation where possible and record
scheduler allocation details.

## Timing fields

- `client_compute_time_sum_sec`: total sequential client execution cost.
- `client_compute_time_max_sec`: synchronous-parallel client latency proxy.
- `server_aggregation_time_sec`: server strategy update time.
- `evaluation_time_sec`: validation and test inference time.
- `round_wall_time_sec`: actual simulator wall-clock time.
- `modeled_comm_time_parallel_sec`: network latency under concurrent
  per-client links.
- `modeled_comm_time_sequential_sec`: sum of all modeled transfers.

## Recommended paper reporting

Keep measured computation energy, modeled computation energy, and modeled
communication energy in separate columns. Never describe a modeled value as a
hardware measurement.
