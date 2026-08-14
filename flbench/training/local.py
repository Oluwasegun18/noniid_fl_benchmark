from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from .types import TrainResult
from .optimizers import make_optimizer

def train_local(global_model,loader,device,cfg,hooks,context=None):
    model=deepcopy(global_model).to(device); optimizer=make_optimizer(model,cfg); criterion=nn.CrossEntropyLoss(); context={} if context is None else context
    context.update({'criterion':criterion,'device':device,'optimizer':optimizer,'global_model':global_model,'local_steps':0})
    hooks.before_training(context); total=0.0; seen=0; model.train()
    for _ in range(int(cfg['training']['local_epochs'])):
        for inputs,targets in loader:
            inputs,targets=inputs.to(device),targets.to(device); optimizer.zero_grad(set_to_none=True); context.update({'inputs':inputs,'targets':targets})
            loss=hooks.compute_loss(model,inputs,targets,context)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite local loss.')
            loss.backward(); hooks.after_backward(model,context); hooks.optimizer_step(model,optimizer,context)
            n=targets.size(0); total += loss.item()*n; seen += n; context['local_steps'] += 1
    aux=hooks.after_training(model,context)
    return TrainResult({k:v.detach().cpu() for k,v in model.state_dict().items()},{'local_loss':total/max(seen,1),'local_steps':float(context['local_steps'])},aux)
