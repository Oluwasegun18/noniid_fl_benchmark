from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from flbench.algorithms.registry import REGISTRY
class Tiny(nn.Module):
    def __init__(self):super().__init__();self.f=nn.Linear(4,8);self.c=nn.Linear(8,3)
    def forward_features(self,x):return torch.relu(self.f(x))
    def forward(self,x,return_features=False):z=self.forward_features(x);y=self.c(z);return (z,y) if return_features else y
def cfg(name):return {'training':{'optimizer':'sgd','learning_rate':0.05,'momentum':0.,'weight_decay':0.,'local_epochs':1},'algorithm':{'name':name,'parameters':{'fedprox_mu':0.01,'scaffold_server_lr':1.,'fednova_server_lr':1.,'feddyn_alpha':0.01,'moon_mu':1.,'moon_temperature':0.5,'fedsam_rho':0.05,'fedgucci_beta':0.1,'fedgucci_num_anchors':2,'fedgucci_lambdas':[0.5]}}}
torch.manual_seed(1);x=torch.randn(32,4);y=torch.randint(0,3,(32,));loader=DataLoader(TensorDataset(x,y),batch_size=8)
for name,cls in REGISTRY.items():
    torch.manual_seed(2);model=Tiny();alg=cls();c=cfg(name);alg.initialize(model,1,c);u=alg.client_update(0,model,loader,torch.device('cpu'),c);alg.server_update(model,[u],c);assert all(torch.isfinite(p).all() for p in model.parameters());print('[OK]',name)
