# Benchmark Run Commands and Instructions

## 1. Dataset partitioning

Use synthetic partitioning for:

- CIFAR-10
- CIFAR-100
- Covtype

Use natural client partitions for:

- FEMNIST: writer-based clients
- Shakespeare: speaking-role/user-based clients

For FEMNIST and Shakespeare, use:

```text
partition.method: natural
```

and do not apply Dirichlet repartitioning.

---

## 2. Case indexing

### Synthetic datasets

CIFAR-10, CIFAR-100, and Covtype use three partition-severity cases across eight algorithms:

```text
3 cases × 8 algorithms = 24 cases
```

Valid case indices:

```text
0–23
```

Check cases with:

```bash
python run_case_index.py --dataset cifar10 --index 0
python run_case_index.py --dataset cifar10 --index 8
python run_case_index.py --dataset cifar10 --index 16
```

Use the same pattern for `cifar100` or `covtype`.

### Natural datasets

FEMNIST and Shakespeare use one natural partition across eight algorithms:

```text
1 natural partition × 8 algorithms = 8 cases
```

Valid case indices:

```text
0–7
```

Check FEMNIST cases:

```bash
python run_case_index.py --dataset femnist --index 0
python run_case_index.py --dataset femnist --index 7
```

Check Shakespeare cases:

```bash
python run_case_index.py --dataset shakespeare --index 0
python run_case_index.py --dataset shakespeare --index 7
```

Algorithm order:

```text
0  fedavg
1  fedprox
2  scaffold
3  fednova
4  feddyn
5  moon
6  fedsam
7  fedgucci
```

---

## 3. One-case Optuna search commands

### CIFAR-10

```bash
python run_optuna_dirichlet_search.py --dataset cifar10 --case high --algorithm fedavg --search-config configs/controlled_search_grid.yaml
```

### CIFAR-100

```bash
python run_optuna_dirichlet_search.py --dataset cifar100 --case high --algorithm fedavg --search-config configs/controlled_search_grid.yaml
```

### Covtype

```bash
python run_optuna_dirichlet_search.py --dataset covtype --case high --algorithm fedavg --search-config configs/controlled_search_grid.yaml
```

### FEMNIST natural partition

```bash
python run_optuna_dirichlet_search.py --dataset femnist --case natural --algorithm fedavg --search-config configs/controlled_search_grid.yaml
```

### Shakespeare natural partition

```bash
python run_optuna_dirichlet_search.py --dataset shakespeare --case natural --algorithm fedavg --search-config configs/controlled_search_grid.yaml
```

Replace `fedavg` with any other supported algorithm as needed.

---

## 4. One-trial smoke test

Before launching a full search on a new machine or dataset, run one trial.

### FEMNIST

```bash
python run_optuna_dirichlet_search.py --dataset femnist --case natural --algorithm fedavg --search-config configs/controlled_search_grid.yaml --n-trials 1
```

### Shakespeare

```bash
python run_optuna_dirichlet_search.py --dataset shakespeare --case natural --algorithm fedavg --search-config configs/controlled_search_grid.yaml --n-trials 1
```

### Covtype

```bash
python run_optuna_dirichlet_search.py --dataset covtype --case high --algorithm fedavg --search-config configs/controlled_search_grid.yaml --n-trials 1
```

Use the smoke test to verify:

- dataset loading
- partitioning
- model creation
- local training
- aggregation
- validation
- Optuna storage
- output generation

before starting the complete grid.

---

## 5. Local desktop runs with the Python runner

Use the cross-platform Python runner:

```text
run_local_optuna.py
```

The same runner can be used on Windows and Ubuntu. This avoids PowerShell execution-policy issues and keeps the local workflow consistent across machines.

### FEMNIST on Windows

Run the full natural-partition search:

```powershell
python run_local_optuna.py --dataset femnist
```

Run only part of the algorithm range:

```powershell
python run_local_optuna.py --dataset femnist --start-index 0 --end-index 3
```

or:

```powershell
python run_local_optuna.py --dataset femnist --start-index 4 --end-index 7
```

### Shakespeare on Windows

Run the full natural-partition search:

```powershell
python run_local_optuna.py --dataset shakespeare
```

Run only part of the algorithm range:

```powershell
python run_local_optuna.py --dataset shakespeare --start-index 0 --end-index 3
```

or:

```powershell
python run_local_optuna.py --dataset shakespeare --start-index 4 --end-index 7
```

### Covtype on Ubuntu

Run the full three-case search:

```bash
python run_local_optuna.py --dataset covtype
```

Run only part of the case range:

```bash
python run_local_optuna.py --dataset covtype --start-index 0 --end-index 7
```

```bash
python run_local_optuna.py --dataset covtype --start-index 8 --end-index 15
```

```bash
python run_local_optuna.py --dataset covtype --start-index 16 --end-index 23
```

### Local smoke test

Before launching a full local search, run one case with one Optuna trial:

```powershell
python run_local_optuna.py --dataset femnist --start-index 0 --end-index 0 --n-trials 1
```

```powershell
python run_local_optuna.py --dataset shakespeare --start-index 0 --end-index 0 --n-trials 1
```

```bash
python run_local_optuna.py --dataset covtype --start-index 0 --end-index 0 --n-trials 1
```

For a single-GPU desktop, run one Optuna case at a time.

On Windows, local Optuna storage should use SQLite rather than Journal storage.

---

## 6. Cluster runs

Create the log directory before submitting:

```bash
mkdir -p logs
```

Submit the CIFAR-10 search:

```bash
sbatch cluster/submit_cifar10_optuna_search.sh
```

For CIFAR-100, use the corresponding CIFAR-100 submission script.

The current cluster configuration runs:

```text
24 total cases
maximum 4 concurrent cases
1 GPU per case
8 CPUs per case
```

Monitor jobs with:

```bash
squeue -u ol_tal
```

Inspect a log with:

```bash
cat logs/optuna_<JOBID>_<ARRAY_INDEX>.out
```

or:

```bash
cat logs/optuna_<JOBID>_<ARRAY_INDEX>.err
```

---

## 7. FEMNIST data requirement

FEMNIST should contain LEAF-style JSON files under one of:

```text
<data_dir>/femnist/train/
<data_dir>/femnist/test/
```

or:

```text
<data_dir>/femnist/data/train/
<data_dir>/femnist/data/test/
```

Verify the files with:

```bash
find data/femnist/data/train -name "*.json" | head
find data/femnist/data/test -name "*.json" | head
```

When using the natural partition, the original writer identities are preserved as clients.

---

## 8. Shakespeare data requirement

Shakespeare should contain LEAF-style JSON files under one of:

```text
<data_dir>/shakespeare/train/
<data_dir>/shakespeare/test/
```

or:

```text
<data_dir>/shakespeare/data/train/
<data_dir>/shakespeare/data/test/
```

Verify the files with:

```bash
find data/shakespeare/data/train -name "*.json" | head
find data/shakespeare/data/test -name "*.json" | head
```

When using the natural partition, the original speaking-role/user identities are preserved as clients.

---

## 9. Search outputs

Each Optuna search should generate files such as:

```text
optuna_trials.csv
search_summary.json
best_config.yaml
best_parameters.yaml
candidate_runs/
```

The selected configuration is taken from the best observed validation result within the predefined search space.

Search-stage energy is descriptive only and is not used for final algorithm energy ranking.

---

## 10. Confirmation runs

After search completion:

1. Load the selected configuration from `best_parameters.yaml` or `best_config.yaml`.
2. Run the predefined confirmation seeds.
3. Use the confirmation runs for final test reporting.
4. Use only valid confirmation-stage measured GPU energy for energy ranking.

Final confirmation reporting should include:

```text
test accuracy
macro-F1
runtime
communication cost
measured GPU energy
```

---

## 11. Recommended execution split

```text
Cluster          -> CIFAR-10 / CIFAR-100 / remaining heavy cases
Windows desktop  -> FEMNIST
Windows desktop  -> Shakespeare
Ubuntu desktop   -> Covtype
```

Prefer dataset-level parallelism across different machines rather than running multiple training processes on one GPU.
