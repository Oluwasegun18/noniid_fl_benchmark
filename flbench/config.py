from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

REQUIRED={'experiment','data','partition','model','training','federation','algorithm','stopping','evaluation','output'}
SUPPORTED_DATASETS={'femnist','cifar10','cifar100','covtype','shakespeare'}
SUPPORTED_ALGORITHMS={'fedavg','fedprox','scaffold','fednova','feddyn','moon','fedsam','fedgucci'}
SUPPORTED_OPTIMIZERS={'sgd','adam','adamw'}

def load_config(path):
    with Path(path).open('r',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    validate_config(cfg); return cfg

def validate_config(cfg):
    missing=REQUIRED-set(cfg)
    if missing: raise ValueError(f'Missing configuration sections: {sorted(missing)}')
    dataset=str(cfg['data']['dataset']).lower()
    if dataset not in SUPPORTED_DATASETS: raise ValueError(f'Unsupported dataset: {dataset}')
    alg=str(cfg['algorithm']['name']).lower()
    if alg not in SUPPORTED_ALGORITHMS: raise ValueError(f'Unsupported algorithm: {alg}')
    opt=str(cfg['training'].get('optimizer','sgd')).lower()
    if opt not in SUPPORTED_OPTIMIZERS: raise ValueError(f'Unsupported optimizer: {opt}')
    if not 0<float(cfg['federation']['participation_rate'])<=1: raise ValueError('participation_rate must be in (0,1].')
    if int(cfg['partition'].get('num_clients',1))<1: raise ValueError('num_clients must be positive.')
    if int(cfg['evaluation'].get('frequency',1))<1: raise ValueError('evaluation.frequency must be >= 1.')

def deep_set(cfg,dotted,value):
    out=deepcopy(cfg); cur=out; keys=dotted.split('.')
    for key in keys[:-1]: cur=cur.setdefault(key,{})
    cur[keys[-1]]=value; validate_config(out); return out
