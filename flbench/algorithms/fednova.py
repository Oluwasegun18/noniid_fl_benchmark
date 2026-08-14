from copy import deepcopy
import torch
from torch import nn
from flbench.training.types import ClientUpdate
from flbench.training.local import make_optimizer
from .base import FederatedAlgorithm

class FedNova(FederatedAlgorithm):
    name='fednova'
    def client_update(self,client_id,global_model,loader,device,cfg):
        model=deepcopy(global_model).to(device); initial={k:v.detach().clone().to(device) for k,v in global_model.state_dict().items()}; opt=make_optimizer(model,cfg); crit=nn.CrossEntropyLoss(); steps=0; total=0; seen=0
        for _ in range(int(cfg['training']['local_epochs'])):
            for x,y in loader:
                x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=crit(model(x),y); loss.backward(); opt.step(); steps+=1; total+=loss.item()*len(y); seen+=len(y)
        state=model.state_dict(); delta={k:((initial[k]-state[k])/steps).cpu() for k in state if state[k].is_floating_point()}
        return ClientUpdate(client_id,len(loader.dataset),{k:v.detach().cpu() for k,v in state.items()},{'local_loss':total/max(seen,1),'local_steps':float(steps)},{'normalized_delta':delta,'tau':float(steps)})
    def server_update(self,model,updates,cfg):
        total=sum(u.num_samples for u in updates); tau=sum((u.num_samples/total)*u.auxiliary['tau'] for u in updates); eta=float(cfg['algorithm']['parameters'].get('fednova_server_lr',1.0)); state=model.state_dict(); new={}
        for k,v in state.items():
            if v.is_floating_point():
                norm=sum((u.num_samples/total)*u.auxiliary['normalized_delta'][k] for u in updates); new[k]=v-eta*tau*norm.to(v.device)
            else:new[k]=v
        model.load_state_dict(new)

    def client_upload_num_bytes(self,update):
        from flbench.experiment.resources import payload_num_bytes
        return payload_num_bytes(update.auxiliary.get('normalized_delta')) + 8
