import torch
from flbench.training.hooks import LocalTrainingHooks
from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin

class FedDynHooks(LocalTrainingHooks):
    def before_training(self,ctx):
        d=ctx['device']; ctx['global_params']={n:p.detach().clone().to(d) for n,p in ctx['global_model'].named_parameters()}; ctx['dual']={n:v.to(d) for n,v in ctx['dual'].items()}
    def compute_loss(self,model,inputs,targets,ctx):
        base=ctx['criterion'](model(inputs),targets); alpha=float(ctx['alpha']); reg=torch.zeros((),device=inputs.device)
        for n,p in model.named_parameters(): reg += -torch.sum(p*ctx['dual'][n]) + 0.5*alpha*torch.sum((p-ctx['global_params'][n])**2)
        return base+reg
    def after_training(self,model,ctx):
        alpha=float(ctx['alpha']); new={n:(ctx['dual'][n]-alpha*(p.detach()-ctx['global_params'][n])).cpu() for n,p in model.named_parameters()}; return {'new_dual':new}
class FedDyn(StandardLocalAlgorithmMixin,FederatedAlgorithm):
    name='feddyn'
    def initialize(self,model,num_clients,cfg):
        super().initialize(model,num_clients,cfg); self.duals={i:{n:torch.zeros_like(p.detach().cpu()) for n,p in model.named_parameters()} for i in range(num_clients)}
    def make_hooks(self,*a): return FedDynHooks()
    def make_context(self,client_id,global_model,cfg): return {'dual':self.duals[client_id],'alpha':float(cfg['algorithm']['parameters']['feddyn_alpha'])}
    def server_update(self,model,updates,cfg):
        for u in updates:self.duals[u.client_id]=u.auxiliary['new_dual']
        self.weighted_average(model,updates)
    def state_dict(self):return {'duals':self.duals}

    def load_state_dict(self,state): self.duals={int(k):v for k,v in state['duals'].items()}
