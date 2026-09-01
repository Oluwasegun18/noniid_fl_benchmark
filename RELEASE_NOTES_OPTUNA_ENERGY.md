# Release update: parallel Optuna search + controlled confirmation energy

Key changes:

- fixed shared CPU/CUDA aggregation mismatch in `flbench/algorithms/base.py`;
- added Optuna GridSampler coordination without changing the predefined exhaustive grid;
- separated parameter search from confirmation execution;
- added 4-GPU/32-CPU SLURM search array for CIFAR-10;
- confirmation array defaults to serialized execution for energy integrity;
- search-stage energy is descriptive only and never used for algorithm energy ranking;
- confirmation uses full-run measured GPU energy as the primary energy metric;
- modeled CPU energy and hybrid totals remain separate/descriptive;
- added CUDA warm-up outside the measured full-run boundary;
- NVML handle resolution now respects `CUDA_VISIBLE_DEVICES` / GPU or MIG UUIDs;
- ranking measurements monitor concurrent GPU compute processes throughout the run;
- confirmation summaries exclude invalid energy trials from energy means;
- 25 tests pass with `python -m pytest -q`.
