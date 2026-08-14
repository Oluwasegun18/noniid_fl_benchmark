from copy import deepcopy
import torch
import torch.nn.functional as F
from flbench.training.hooks import LocalTrainingHooks
from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin

class MoonHooks(LocalTrainingHooks):
    def before_training(self,ctx):
        d=ctx['device']; ctx['global_ref']=deepcopy(ctx['global_model']).to(d).eval(); ctx['previous']=deepcopy(ctx['global_model']).to(d); ctx['previous'].load_state_dict(ctx['previous_state']); ctx['previous'].eval()
        for m in (ctx['global_ref'],ctx['previous']):
            for p in m.parameters():p.requires_grad_(False)
    def compute_loss(self,model,inputs,targets,ctx):
        z,logits=model(inputs,return_features=True)
        with torch.no_grad(): zg,_=ctx['global_ref'](inputs,return_features=True); zp,_=ctx['previous'](inputs,return_features=True)
        contrast=torch.stack([F.cosine_similarity(z,zg,dim=1),F.cosine_similarity(z,zp,dim=1)],1)/float(ctx['temperature']); zeros=torch.zeros(inputs.size(0),dtype=torch.long,device=inputs.device)
        return ctx['criterion'](logits,targets)+float(ctx['mu'])*ctx['criterion'](contrast,zeros)
    def after_training(self,model,ctx):return {'previous_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
class MOON(StandardLocalAlgorithmMixin,FederatedAlgorithm):
    name='moon'
    def initialize(self,model,num_clients,cfg):
        super().initialize(model,num_clients,cfg); initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; self.previous={i:{k:v.clone() for k,v in initial.items()} for i in range(num_clients)}
    def make_hooks(self,*a):return MoonHooks()
    def make_context(self,client_id,global_model,cfg):
        p=cfg['algorithm']['parameters']; return {'previous_state':self.previous[client_id],'mu':float(p['moon_mu']),'temperature':float(p['moon_temperature'])}
    def server_update(self,model,updates,cfg):
        for u in updates:self.previous[u.client_id]=u.auxiliary['previous_state']
        self.weighted_average(model,updates)
