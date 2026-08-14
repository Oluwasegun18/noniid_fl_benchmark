from __future__ import annotations
import hashlib, json, re

def _fmt(value) -> str:
    text=str(value).replace('.','p').replace('-','m')
    return re.sub(r'[^A-Za-z0-9_]+','',text)

def experiment_id(cfg: dict) -> str:
    alg=cfg['algorithm']['name'].lower(); p=cfg['partition']; f=cfg['federation']; t=cfg['training']; e=cfg['experiment']; dataset=cfg['data']['dataset'].lower(); model=str(cfg['model'].get('architecture','auto')).lower(); optimizer=str(t.get('optimizer','sgd')).lower()
    if str(p.get('method','')).lower()=='natural':
        heterogeneity='natural'; nclients='Nnatural'
    else:
        heterogeneity=f"{_fmt(p.get('method'))}-a{_fmt(p.get('alpha'))}"; nclients=f"N{p.get('num_clients')}"
    readable=(f"{dataset}-{heterogeneity}-{nclients}-C{_fmt(f['participation_rate'])}-E{t['local_epochs']}-B{t['batch_size']}-lr{_fmt(t['learning_rate'])}-{optimizer}-{model}-{alg}-s{e['seed']}-t{e.get('trial',1)}")
    digest=hashlib.sha1(json.dumps(cfg,sort_keys=True,default=str).encode('utf-8')).hexdigest()[:8]
    return f"{readable}__{digest}"
