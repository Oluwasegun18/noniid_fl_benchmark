from collections import deque
import torch
from torch.func import functional_call
from flbench.training.hooks import LocalTrainingHooks
from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin

class GuCciHooks(LocalTrainingHooks):
    def before_training(self,ctx):
        d=ctx['device']; ctx['anchors']=[{k:v.to(d) for k,v in s.items()} for s in ctx['anchor_states']]
    def compute_loss(self,model,inputs,targets,ctx):
        criterion=ctx['criterion']; base=criterion(model(inputs),targets); params=dict(model.named_parameters()); buffers=dict(model.named_buffers()); names=set(params); total=torch.zeros((),device=inputs.device); terms=0
        for anchor in ctx['anchors']:
            for lam in ctx['lambdas']:
                mixed={n:float(lam)*params[n]+(1-float(lam))*anchor[n] for n in names}; total += criterion(functional_call(model,(mixed,buffers),(inputs,)),targets); terms+=1
        return base + float(ctx['beta'])*(total/terms if terms else total)

class FedGuCci(StandardLocalAlgorithmMixin,FederatedAlgorithm):
    name='fedgucci'
    def initialize(self,model,num_clients,cfg):
        super().initialize(model,num_clients,cfg)
        n=int(cfg['algorithm']['parameters']['fedgucci_num_anchors'])
        # Official rule: at round t, use the most recent N broadcast global
        # models; before N are available, use all available global models.
        self.history=deque(maxlen=n)
        self.history.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    def make_hooks(self,*a): return GuCciHooks()
    def make_context(self,client_id,global_model,cfg):
        p=cfg['algorithm']['parameters']; lambdas=p.get('fedgucci_lambdas')
        if lambdas is None:
            samples=int(p.get('fedgucci_interpolation_samples',1)); generator=torch.Generator().manual_seed(int(cfg['experiment']['seed'])+client_id); lambdas=torch.rand(samples,generator=generator).tolist()
        return {'anchor_states':list(self.history),'beta':float(p['fedgucci_beta']),'lambdas':lambdas}
    def server_update(self,model,updates,cfg):
        self.weighted_average(model,updates)
        # The newly generated global model becomes the current broadcast anchor
        # for the following round; deque automatically retains only recent N.
        self.history.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    def state_dict(self): return {'history':list(self.history),'maxlen':self.history.maxlen}
    def load_state_dict(self,state): self.history=deque(state['history'],maxlen=int(state.get('maxlen',len(state['history']) or 1)))
