# Convergence-based controlled search

## Termination rule

Both candidate search runs and confirmation runs terminate when validation
accuracy shows no observable improvement.  The default rule is:

- validation evaluation every 5 communication rounds;
- minimum observable improvement (`min_delta`) = 0.0001;
- patience = 10 validation evaluations;
- no convergence stop before round 50;
- 1000 rounds as a safety ceiling only.

Thus, after round 50, a run normally stops after 10 consecutive validation
checks (50 communication rounds at the default evaluation frequency) without
an improvement larger than 0.0001.

The best validation model is saved immediately to `best_validation_model.pt`.
At termination, the runner restores that model and recomputes the final
validation and test metrics.  Candidate selection therefore uses the strongest
observed validation point rather than the metric at the last plateau round.

## Primary shared grid

- learning rate: 0.01, 0.04, 0.1
- local epochs: 1, 3, 5, 10
- batch size: 32, 64, 128
- optimizer: SGD (fixed)

## Algorithm-specific grid

- FedAvg: none
- FedProx: mu = 0.001, 0.01, 0.1, 1.0
- SCAFFOLD: server LR fixed at 1.0; local LR is covered by the shared grid
- FedNova: normalization fixed; no artificial server-LR search axis
- FedDyn: alpha = 0.001, 0.01, 0.1
- MOON: mu = 0.1, 1, 5, 10; temperature fixed at 0.5
- FedSAM: rho = 0.1, 0.5, 1.0
- FedGuCci: beta = 0.025, 0.1, 0.4, 1.6; anchors = 1, 3, 5, 9

## Cluster launch

The existing launch interface is unchanged:

```bash
python show_search_grid.py
python run_dirichlet_search.py --list
sbatch cluster/submit_dirichlet_grid_array.sh
```

To start with CIFAR-10 only, submit the three cases individually:

```bash
sbatch --export=ALL,DATASET=cifar10,CASE=high cluster/run_single_dirichlet_case.sh
sbatch --export=ALL,DATASET=cifar10,CASE=mild cluster/run_single_dirichlet_case.sh
sbatch --export=ALL,DATASET=cifar10,CASE=iid cluster/run_single_dirichlet_case.sh
```
