# Optuna parallel search and confirmation-energy policy

## Scientific split

The benchmark now separates two execution stages:

1. **Search (throughput-oriented)**: Optuna coordinates the predefined parameter search. Search energy is retained as descriptive search cost only; it is not used to rank algorithm energy efficiency.
2. **Confirmation (measurement-controlled)**: the selected configuration is rerun on independent confirmation seeds. Measured run-level GPU energy is the primary energy-ranking quantity. Hybrid measured+modeled energy is descriptive only.

## Search space

The default `configs/controlled_search_grid.yaml` keeps the paper-informed exhaustive grid:

- learning rate: `{0.01, 0.04, 0.1}`
- local epochs: `{1, 3, 5, 10}`
- batch size: `{32, 64, 128}`
- FedProx mu: `{0.001, 0.01, 0.1, 1}`
- FedDyn alpha: `{0.001, 0.01, 0.1}`
- MOON mu: `{0.1, 1, 5, 10}`, temperature fixed at `0.5`
- FedSAM rho: `{0.1, 0.5, 1.0}`
- FedGuCci beta: `{0.025, 0.1, 0.4, 1.6}`, anchors `{1, 3, 5, 9}`
- SCAFFOLD global step: fixed at `1`
- FedNova normalization: fixed by the algorithm

Optuna uses `GridSampler` by default, so introducing Optuna does **not** change the exhaustive-search methodology. It adds resumability and worker coordination.

## Cluster throughput with the 32-CPU / 4-GPU quota

Before submission:

```bash
mkdir -p logs
```

For CIFAR-10, all 3 Dirichlet cases x 8 algorithms are represented by 24 array tasks. At most four run at once:

```bash
sbatch cluster/submit_cifar10_optuna_search.sh
```

The script requests 8 CPUs + 1 GPU per task and uses `%4`, so the maximum concurrent request is 32 CPUs and 4 GPUs.

### Multiple Optuna workers for one large case

For a very large study such as FedGuCci, multiple workers can share one Optuna study:

```bash
export DATASET=cifar10
export CASE=high
export ALGORITHM=fedgucci
sbatch cluster/run_parallel_optuna_workers_one_case.sh
```

For multi-node workers, prefer a shared PostgreSQL/MySQL storage URL:

```bash
export OPTUNA_STORAGE_URL='postgresql+psycopg2://user:password@host/dbname'
```

Without `OPTUNA_STORAGE_URL`, the code uses Optuna Journal storage in the algorithm output directory. Use this only when the cluster filesystem provides reliable shared file locking.

## Confirmation

Run confirmation only after `best_parameters.yaml` exists for the case:

```bash
python run_confirmation.py --dataset cifar10 --case high --algorithm fedavg
```

Or submit all CIFAR-10 confirmation cases:

```bash
sbatch cluster/submit_cifar10_confirmation.sh
```

The confirmation array defaults to `%1`, deliberately serializing cases to maximize measurement integrity. Change it to `%4` only after confirming that each allocated GPU/MIG instance has an independently valid energy measurement scope.

## Energy fields

Each completed run summary now includes:

- `run_energy_primary_j`: measured GPU energy used as the candidate primary energy quantity.
- `run_gpu_energy_measured_j`: same measured GPU component explicitly named.
- `run_cpu_energy_measured_j`: hardware CPU energy when RAPL is accessible.
- `run_cpu_energy_modeled_j`: modeled CPU component when direct CPU measurement is unavailable.
- `run_total_energy_hybrid_j`: measured components plus modeled fallback; descriptive only.
- `run_energy_valid_for_ranking`: whether the confirmation GPU energy passed the isolation checks.
- `run_energy_invalid_reason`: reason the energy should not be used for ranking.
- `run_gpu_counter_scope`: `physical_gpu`, `mig_device`, or `unavailable`.
- `run_gpu_uuid`: NVML GPU/MIG UUID when available.
- `run_concurrent_gpu_processes_detected`: whether another GPU compute PID was observed during the measurement interval.

A background isolation monitor checks for foreign GPU compute processes throughout ranking-role measurements, not only at the start/end.

`confirmation_summary.json` computes energy means only from confirmation trials with `run_energy_valid_for_ranking=true`. Accuracy/runtime remain reported even when an energy reading is invalid.

## Run-level measurement boundary

Before the full training-run energy counter starts, a short inference-only CUDA warm-up is performed. The primary run-level measurement then covers:

- client local training,
- server aggregation,
- periodic validation/evaluation used by the convergence controller.

The measurement stops before restoration of the best checkpoint and final reporting evaluation. This avoids assigning one-time CUDA initialization to the first client and avoids summing overlapping nested energy trackers for the primary ranking quantity.

## Aggregation device fix

`FederatedAlgorithm.weighted_average()` now explicitly transfers stored client tensors to the global model's device and dtype before aggregation. Client updates may remain on CPU between local training and aggregation, while arithmetic is performed on the correct GPU device. This resolves the observed `cuda:0` versus `cpu` aggregation failure across FedAvg/SCAFFOLD and other methods using the shared averaging path.
