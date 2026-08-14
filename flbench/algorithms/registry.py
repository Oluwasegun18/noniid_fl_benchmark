from .fedavg import FedAvg
from .fedprox import FedProx
from .scaffold import SCAFFOLD
from .fednova import FedNova
from .feddyn import FedDyn
from .moon import MOON
from .fedsam import FedSAM
from .fedgucci import FedGuCci
REGISTRY={'fedavg':FedAvg,'fedprox':FedProx,'scaffold':SCAFFOLD,'fednova':FedNova,'feddyn':FedDyn,'moon':MOON,'fedsam':FedSAM,'fedgucci':FedGuCci}
def build_algorithm(cfg):
    name=cfg['algorithm']['name'].lower()
    if name not in REGISTRY:raise ValueError(f'Unknown algorithm: {name}')
    return REGISTRY[name]()
