from __future__ import annotations
import torch

def make_optimizer(model,cfg):
    t=cfg['training']; name=str(t.get('optimizer','sgd')).lower(); lr=float(t['learning_rate']); wd=float(t.get('weight_decay',0.0)); extra=t.get('optimizer_parameters',{}) or {}
    if name=='sgd':
        return torch.optim.SGD(model.parameters(),lr=lr,momentum=float(extra.get('momentum',t.get('momentum',0.0))),weight_decay=wd,nesterov=bool(extra.get('nesterov',False)))
    if name=='adam':
        return torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd,betas=tuple(extra.get('betas',[0.9,0.999])),eps=float(extra.get('eps',1e-8)))
    if name=='adamw':
        return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=tuple(extra.get('betas',[0.9,0.999])),eps=float(extra.get('eps',1e-8)))
    raise ValueError(f'Unsupported optimizer: {name}. Supported: sgd, adam, adamw.')
