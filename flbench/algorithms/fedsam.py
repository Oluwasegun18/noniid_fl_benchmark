import torch
from flbench.training.hooks import LocalTrainingHooks
from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin
class SAMHooks(LocalTrainingHooks):
    def optimizer_step(self,model,optimizer,ctx):
        rho=float(ctx['rho']); grads=[p.grad for p in model.parameters() if p.grad is not None]; norm=torch.norm(torch.stack([g.norm(2) for g in grads]),2); perturb=[]
        with torch.no_grad():
            scale=rho/(norm+1e-12)
            for p in model.parameters():
                e=None if p.grad is None else p.grad*scale
                if e is not None:p.add_(e)
                perturb.append(e)
        optimizer.zero_grad(set_to_none=True); second=ctx['criterion'](model(ctx['inputs']),ctx['targets']); second.backward()
        with torch.no_grad():
            for p,e in zip(model.parameters(),perturb):
                if e is not None:p.sub_(e)
        optimizer.step()
class FedSAM(StandardLocalAlgorithmMixin,FederatedAlgorithm):
    name='fedsam'
    def make_hooks(self,*a):return SAMHooks()
    def make_context(self,client_id,global_model,cfg):return {'rho':float(cfg['algorithm']['parameters']['fedsam_rho'])}
