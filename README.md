# Unified Non-IID Federated Learning Benchmark

A reusable, framework-independent benchmark for CIFAR-10 and non-IID federated learning.
The code uses algorithm logic inspired by canonical implementations, while all algorithms share:

- configuration loading
- dataset preparation and partitioning
- model construction and initialization
- client sampling
- local training engine
- evaluation
- stopping rules
- experiment execution
- logging and checkpointing

Implemented algorithms:

- FedAvg
- FedProx
- SCAFFOLD
- FedNova
- FedDyn
- MOON
- FedSAM
- FedGuCci

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Validate without downloading CIFAR-10

```bash
python validate_algorithms.py
pytest -q
```

## Run one experiment

```bash
python run_experiment.py --config configs/common_protocol.yaml --algorithm fedavg
```

## Run a matched algorithm sweep

```bash
python run_sweep.py --config configs/common_protocol.yaml --grid configs/algorithm_grid.yaml
```

## Design principle

Algorithm modules contain only genuine mathematical or state-management differences.
They do not implement their own dataset loaders, experiment runners, evaluators, or logging systems.


## Descriptive output names and resource tracking

Output directories now use a readable identifier, for example:

```text
cifar10-dirichlet-a0p5-N10-C1p0-E5-B64-lr0p01-fedavg-s1-t1__2f71a3c9
```

The suffix is a short hash of the complete resolved configuration. This keeps
the name readable while preventing collisions when less visible parameters differ.

`round_metrics.csv` now records compute timing, communication volume, modeled
communication time and energy, and measured CPU/GPU energy when hardware counters
are available. Missing hardware measurements are stored as empty values rather
than replaced with undocumented estimates.


## Cross-platform computation and communication tracking

This version records resource usage at three levels:

- `client_metrics.csv`: one record per selected client and round;
- `round_metrics.csv`: aggregated client, server, evaluation, and network costs;
- `system_metadata.json`: operating system, processor, memory, PyTorch, CUDA,
  and selected device.

### Windows CPU smoke test

```bash
python run_experiment.py --config configs/windows_cpu_smoke.yaml
```

Windows normally cannot expose Linux RAPL energy counters. The code therefore
uses the configured CPU power model and reports:

```text
energy_source = cpu_modeled:process_utilization
energy_is_estimated = true
```

The measured CPU/GPU fields remain empty, while
`compute_energy_modeled_j` and `compute_energy_reported_j` are populated.

### Linux GPU cluster

Use the same configuration with `experiment.device: auto` or `cuda`.
The tracker attempts, in order:

1. Linux RAPL-compatible package energy for CPU;
2. NVML cumulative total-energy counters for GPU;
3. NVML power sampling and numerical integration;
4. the configured CPU energy model when RAPL is unavailable.

Hardware support and permissions differ by cluster. Always inspect
`energy_source` and `energy_is_estimated`.

### Important interpretation

- `*_energy_measured_j` values come from hardware counters or integrated
  hardware power samples.
- `*_energy_modeled_j` values are estimates based on the configured CPU model.
- `*_energy_reported_j` is the value selected for reporting and can combine
  measured GPU energy with modeled CPU energy.
- `modeled_comm_energy_j` is always a network model, not a direct measurement.
- `round_total_energy_reported_j` combines computation energy with modeled
  communication energy.

The default communication energy coefficients are zero. Set them only after
selecting a network technology and defensible coefficients.


## Controlled-search evaluation with an equal parameter budget

The codebase now supports a second evaluation mode in addition to the common
protocol. Each algorithm receives the same maximum number of candidate
configurations, search rounds, search seeds, validation criterion, and optional
wall-clock budget. Shared training parameters are combined with
algorithm-specific parameters.

Run a small search:

```bash
python run_controlled_search.py \
  --base-config configs/smoke_test.yaml \
  --search-config configs/controlled_search_smoke.yaml
```

Run the full planned search:

```bash
python run_controlled_search.py \
  --base-config configs/common_protocol.yaml \
  --search-config configs/controlled_search_full.yaml
```

For each algorithm the search produces `search_manifest.json`,
`candidate_results.csv`, `best_config.yaml`, `confirmation_results.json`, and
`search_summary.json`. Candidate selection uses validation performance only.
The selected configuration is rerun over the confirmation seeds, and the final
summary reports mean and standard deviation of confirmation test accuracy.

The correct interpretation is the **best observed configuration within the
predefined search space and budget**, not an absolute optimum.


## Windows controlled-search path fix

Controlled-search runs use compact exact directories:

```text
search_outputs/<scenario>/<algorithm>/candidate_runs/c0001/s1/
```

The descriptive experiment identifier remains inside `summary.json`,
`status.json`, and the CSV logs, but it is no longer appended to the filesystem
path. This avoids Windows path-length failures in projects stored under deep
OneDrive or user-profile directories.

Old failed search output can be removed before restarting:

```powershell
Remove-Item -Recurse -Force search_outputs
```


## Extended benchmark release

This release adds a unified dataset interface for FEMNIST, CIFAR-10, CIFAR-100, Covtype, and Shakespeare; dataset-specific models; SGD/Adam/AdamW optimizer factory; checkpoint resumption with algorithm, client-sampling, loader-generator, and stopping state; configured evaluation frequency; algorithm-aware communication accounting; official historical-global-model FedGuCci anchors; and confirmation runtime mean ± sample SD. FEMNIST and Shakespeare use LEAF-style JSON files placed under `data/femnist` and `data/shakespeare`.

### Dataset layout

- **CIFAR-10/CIFAR-100:** downloaded automatically by `torchvision` under `data.data_dir`.
- **Covtype:** downloaded automatically by `scikit-learn` on first use and standardized using statistics fitted only on the training split.
- **FEMNIST:** expects LEAF-style JSON shards under `data/femnist/train` and `data/femnist/test` (or `data/femnist/data/{train,test}`). Writer IDs define natural FL clients.
- **Shakespeare:** expects LEAF-style JSON shards under `data/shakespeare/train` and `data/shakespeare/test` (or `data/shakespeare/data/{train,test}`). LEAF user/role IDs define natural FL clients.

Dataset-specific example configurations are provided in `configs/cifar100.yaml`, `configs/covtype.yaml`, `configs/femnist.yaml`, and `configs/shakespeare.yaml`.


## Exhaustive Dirichlet search matrix

The first full search uses the same Dirichlet partition family for three
severity cases on every dataset:

- **highly non-IID:** alpha = 0.1
- **mildly non-IID:** alpha = 0.5
- **IID-like:** alpha = 100

The alpha=100 case is intentionally called IID-like because it is still a
Dirichlet allocation rather than an exact equal IID split.

The controlled search now evaluates the complete Cartesian grid for each
algorithm. There is no equal `max_configurations_per_algorithm` restriction.
An algorithm with more method-specific hyperparameters naturally has a larger
grid. The grid size and total search runtime are saved in each search summary.

Inspect the setup:

```bash
python show_search_grid.py
python run_dirichlet_search.py --list
```

Run one scenario:

```bash
python run_dirichlet_search.py --dataset cifar10 --case high
python run_dirichlet_search.py --dataset cifar10 --case mild
python run_dirichlet_search.py --dataset cifar10 --case iid
```

Submit all 15 dataset/severity scenarios on SLURM:

```bash
sbatch cluster/submit_dirichlet_grid_array.sh
```

Submit only one scenario:

```bash
sbatch --export=ALL,DATASET=cifar10,CASE=high \
  cluster/run_single_dirichlet_case.sh
```

FEMNIST and Shakespeare continue to be loaded from LEAF-style source files,
but for this experiment their flattened training examples are repartitioned
with the same Dirichlet procedure as the other datasets. Their natural
writer/speaking-role partitions remain available for later experiments.

On a cluster without internet access, stage CIFAR-10, CIFAR-100, and Covtype
before submitting compute jobs. FEMNIST and Shakespeare should be placed under
`data/femnist` and `data/shakespeare`.


## Convergence-based production search

See `CONVERGENCE_SEARCH_GUIDE.md`. Search and confirmation now use the same validation-convergence rule and restore the best-validation checkpoint before final testing. The primary grid is paper-informed and exhaustive per algorithm.

## Parallel search / confirmation update

The recommended workflow is now **Optuna search -> selected configuration -> independent confirmation**. Search energy is descriptive only; confirmation-stage measured GPU energy is the only primary energy-ranking quantity. See `OPTUNA_PARALLEL_ENERGY_GUIDE.md` and use `cluster/submit_cifar10_optuna_search.sh` followed by `cluster/submit_cifar10_confirmation.sh`.
