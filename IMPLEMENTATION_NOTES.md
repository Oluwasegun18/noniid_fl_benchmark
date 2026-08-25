# Implementation notes

This codebase intentionally does not reproduce the Flower repository structure. Flower Baselines are used only as references for algorithm logic. Shared infrastructure is used for configuration, dataset preparation, model construction, local training, experiment execution, evaluation, resource tracking, and checkpointing.

Algorithm modules contain only the mathematical or state-management differences required by each method, including objective modifications, gradient corrections, optimizer-step overrides, server aggregation differences, persistent client/server state, and algorithm-specific communication payloads.

## FedGuCci fidelity

FedGuCci uses the official historical-global-model anchor rule: the anchor set contains the most recent `N` global models, and during the initial rounds it contains all global models available so far. The current implementation retains those global-model states locally in a fixed-length history and averages connectivity loss across the retained anchors. Historical anchors are not counted as additional per-round network payloads because the method treats them as previously broadcast global models retained by clients.

## Additional datasets

CIFAR-10 and CIFAR-100 use torchvision; Covtype uses scikit-learn; FEMNIST and Shakespeare use LEAF-style JSON files with natural user/writer partitions. The unified dataset interface returns the datasets, client indices, task metadata, input shape, class/vocabulary size, and optional client names.

## Checkpoint resume

Checkpoints preserve the global model, algorithm-specific state, Python client-sampling RNG state, client DataLoader generator states, stopping-controller state, elapsed runtime, and resolved configuration. This is intended to make cluster job resumption reproduce the sampling and batch-order state of the interrupted run.

### Smoothed convergence controller

`flbench/experiment/stopping.py` uses a moving-average convergence statistic for
`mode: convergence`. The default controlled-search configuration is:

```yaml
termination:
  mode: convergence
  monitor: val_accuracy
  smoothing_window: 3
  min_delta: 0.001
  patience_evaluations: 10
  min_rounds: 50
  evaluation_frequency: 5
  max_search_rounds: 1000
  max_confirmation_rounds: 1000
```

Smoothing affects termination only. Raw validation accuracy controls the saved
best-model checkpoint. Checkpoint/resume state includes the moving window,
smoothed best, raw best, and patience counter, so a resumed job preserves the
same convergence trajectory.
