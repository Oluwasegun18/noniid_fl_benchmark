import argparse,itertools,yaml,traceback
from flbench.config import load_config,deep_set
from flbench.experiment.runner import run_experiment
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--grid',required=True);a=p.parse_args();base=load_config(a.config);grid=yaml.safe_load(open(a.grid))['grid'];keys=list(grid);comb=list(itertools.product(*[grid[k] for k in keys]));print(f'{len(comb)} runs')
for vals in comb:
    cfg=base
    for k,v in zip(keys,vals):cfg=deep_set(cfg,k,v)
    try:print(run_experiment(cfg))
    except Exception:traceback.print_exc()
