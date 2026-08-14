import torch
from flbench.training.hooks import LocalTrainingHooks
from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin

class FedProxHooks(LocalTrainingHooks):
    def before_training(self,ctx):
        device=ctx['device']; ctx['reference']={n:p.detach().clone().to(device) for n,p in ctx['global_model'].named_parameters()}
    def compute_loss(self,model,inputs,targets,ctx):
        base=ctx['criterion'](model(inputs),targets); mu=float(ctx['mu']); prox=sum(((p-ctx['reference'][n])**2).sum() for n,p in model.named_parameters())
        return base+0.5*mu*prox
class FedProx(StandardLocalAlgorithmMixin,FederatedAlgorithm):
    name='fedprox'
    def make_hooks(self,*a): return FedProxHooks()
    def make_context(self,client_id,global_model,cfg): return {'client_id':client_id,'mu':float(cfg['algorithm']['parameters']['fedprox_mu'])}
