from copy import deepcopy
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from flbench.data.partitioning import dirichlet_partition
from flbench.algorithms.fedavg import FedAvg
from flbench.algorithms.fedprox import FedProx
class M(nn.Module):
    def __init__(self):super().__init__();self.l=nn.Linear(4,2)
    def forward(self,x):return self.l(x)
def config(name,mu=0):return {'training':{'optimizer':'sgd','learning_rate':0.1,'momentum':0.,'weight_decay':0.,'local_epochs':1},'algorithm':{'name':name,'parameters':{'fedprox_mu':mu}}}
def test_partition_reproducible():
    y=np.repeat(np.arange(10),100);assert dirichlet_partition(y,5,.5,3,20)==dirichlet_partition(y,5,.5,3,20)
def test_fedprox_zero_equals_fedavg():
    torch.manual_seed(3);base=M();a=deepcopy(base);b=deepcopy(base);x=torch.randn(16,4);y=torch.randint(0,2,(16,));l=DataLoader(TensorDataset(x,y),batch_size=4,shuffle=False);fa,Fp=FedAvg(),FedProx();fa.initialize(a,1,config('fedavg'));Fp.initialize(b,1,config('fedprox',0));ua=fa.client_update(0,a,l,torch.device('cpu'),config('fedavg'));ub=Fp.client_update(0,b,l,torch.device('cpu'),config('fedprox',0));fa.server_update(a,[ua],config('fedavg'));Fp.server_update(b,[ub],config('fedprox',0));assert max((a.state_dict()[k]-b.state_dict()[k]).abs().max().item() for k in a.state_dict())<1e-7
