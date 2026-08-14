from pathlib import Path
import json
import torch
from torch.utils.data import TensorDataset
from flbench.training.optimizers import make_optimizer
from flbench.models.factory import build_model
from flbench.data.datasets import build_dataset
from flbench.algorithms.scaffold import SCAFFOLD
from flbench.algorithms.fednova import FedNova
from flbench.training.types import ClientUpdate


def cfg(dataset='covtype',arch='tabular_mlp'):
    return {'experiment':{'seed':1,'device':'cpu','deterministic':True},'data':{'dataset':dataset,'data_dir':'data','validation_fraction':0.1,'augmentation':False,'subset_size':100},'partition':{'method':'dirichlet','alpha':0.5,'num_clients':2,'seed':1,'cache_dir':'partition_cache','min_samples_per_client':1},'model':{'architecture':arch,'num_classes':7,'initialization_seed':1,'input_dim':54},'training':{'optimizer':'adam','learning_rate':0.001,'momentum':0.0,'weight_decay':0.0,'local_epochs':1,'batch_size':8,'num_workers':0},'federation':{'participation_rate':1.0,'communication_rounds':1,'client_sampling_seed':1},'algorithm':{'name':'fedavg','parameters':{}},'stopping':{'mode':'fixed_rounds'},'evaluation':{'frequency':2},'output':{'save_checkpoints':False,'resume':False}}


def test_optimizer_factory_adam():
    model=torch.nn.Linear(4,2); c=cfg(); opt=make_optimizer(model,c); assert isinstance(opt,torch.optim.Adam)


def test_added_models_construct():
    cases=[('femnist','femnist_cnn',62,(1,28,28)),('cifar100','resnet20',100,(3,32,32)),('covtype','tabular_mlp',7,(54,)),('shakespeare','shakespeare_lstm',80,(80,))]
    class B: pass
    for ds,arch,nc,shape in cases:
        c=cfg(ds,arch); c['model']['num_classes']=nc; b=B(); b.num_classes=nc; b.input_shape=shape
        model=build_model(c,b); assert sum(p.numel() for p in model.parameters())>0


def test_scaffold_counts_control_payload():
    model=torch.nn.Linear(4,2); c=cfg(); c['algorithm']={'name':'scaffold','parameters':{'scaffold_server_lr':1.0}}; alg=SCAFFOLD(); alg.initialize(model,2,c)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}; delta={n:torch.zeros_like(p) for n,p in model.named_parameters()}; update=ClientUpdate(0,1,state,{}, {'control_delta':delta})
    assert alg.client_upload_num_bytes(update) > sum(v.numel()*v.element_size() for v in state.values())


def test_fednova_upload_uses_normalized_delta():
    model=torch.nn.Linear(4,2); state={k:v.detach().cpu() for k,v in model.state_dict().items()}; delta={k:v.clone() for k,v in state.items() if v.is_floating_point()}; update=ClientUpdate(0,1,state,{}, {'normalized_delta':delta,'tau':1.0}); alg=FedNova(); assert alg.client_upload_num_bytes(update)>0
