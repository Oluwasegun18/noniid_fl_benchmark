import argparse
from flbench.config import load_config,deep_set
from flbench.experiment.runner import run_experiment
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--algorithm');a=p.parse_args();cfg=load_config(a.config)
if a.algorithm:cfg=deep_set(cfg,'algorithm.name',a.algorithm)
print(run_experiment(cfg))
