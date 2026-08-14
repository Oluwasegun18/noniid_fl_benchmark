import torch
from flbench.training.hooks import LocalTrainingHooks
from flbench.experiment.resources import payload_num_bytes, state_dict_num_bytes
from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin

class ScaffoldHooks(LocalTrainingHooks):
    def before_training(self,ctx):
        d=ctx['device']; ctx['global_params']={n:p.detach().clone().to(d) for n,p in ctx['global_model'].named_parameters()}; ctx['ci']={n:v.to(d) for n,v in ctx['client_control'].items()}; ctx['c']={n:v.to(d) for n,v in ctx['server_control'].items()}
    def after_backward(self,model,ctx):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.grad is not None: p.grad.add_(ctx['c'][n]-ctx['ci'][n])
    def after_training(self,model,ctx):
        steps=ctx['local_steps']; lr=float(ctx['learning_rate']); new={}; delta={}
        if steps <= 0: raise RuntimeError('SCAFFOLD completed zero local steps.')
        with torch.no_grad():
            for n,p in model.named_parameters():
                value=ctx['ci'][n]-ctx['c'][n]+(ctx['global_params'][n]-p.detach())/(steps*lr); new[n]=value.cpu(); delta[n]=(new[n]-ctx['client_control'][n])
        return {'new_client_control':new,'control_delta':delta}

class SCAFFOLD(StandardLocalAlgorithmMixin,FederatedAlgorithm):
    name='scaffold'
    def initialize(self,model,num_clients,cfg):
        super().initialize(model,num_clients,cfg); zero={n:torch.zeros_like(p.detach().cpu()) for n,p in model.named_parameters()}; self.server_control={n:v.clone() for n,v in zero.items()}; self.client_controls={i:{n:v.clone() for n,v in zero.items()} for i in range(num_clients)}
    def make_hooks(self,*a): return ScaffoldHooks()
    def make_context(self,client_id,global_model,cfg): return {'client_id':client_id,'client_control':self.client_controls[client_id],'server_control':self.server_control,'learning_rate':float(cfg['training']['learning_rate'])}
    def server_update(self,model,updates,cfg):
        self.weighted_average(model,updates)
        for u in updates: self.client_controls[u.client_id]=u.auxiliary['new_client_control']
        eta=float(cfg['algorithm']['parameters'].get('scaffold_server_lr',1.0))
        for n in self.server_control: self.server_control[n] += eta*torch.stack([u.auxiliary['control_delta'][n] for u in updates]).sum(0)/self.num_clients
    def client_upload_num_bytes(self,update): return state_dict_num_bytes(update.state_dict)+payload_num_bytes(update.auxiliary.get('control_delta'))
    def download_num_bytes(self,global_model,selected_clients): return (state_dict_num_bytes(global_model.state_dict())+payload_num_bytes(self.server_control))*int(selected_clients)
    def state_dict(self): return {'server_control':self.server_control,'client_controls':self.client_controls}
    def load_state_dict(self,state): self.server_control=state['server_control']; self.client_controls={int(k):v for k,v in state['client_controls'].items()}
