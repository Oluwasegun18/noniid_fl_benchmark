from .base import FederatedAlgorithm
from .common import StandardLocalAlgorithmMixin
class FedAvg(StandardLocalAlgorithmMixin,FederatedAlgorithm): name='fedavg'
