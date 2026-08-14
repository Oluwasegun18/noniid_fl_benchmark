from .cifar10 import SimpleCIFAR10CNN
from .femnist import FEMNISTCNN
from .resnet20 import ResNet20
from .tabular import TabularMLP
from .shakespeare import ShakespeareLSTM

DEFAULT_ARCHITECTURES={'cifar10':'simple_cnn','cifar100':'resnet20','femnist':'femnist_cnn','covtype':'tabular_mlp','shakespeare':'shakespeare_lstm'}

def build_model(cfg, bundle=None):
    dataset=str(cfg['data']['dataset']).lower(); arch=str(cfg['model'].get('architecture','auto')).lower()
    if arch=='auto': arch=DEFAULT_ARCHITECTURES[dataset]
    classes=int(cfg['model'].get('num_classes') or (bundle.num_classes if bundle else 0))
    if arch=='simple_cnn': return SimpleCIFAR10CNN(classes)
    if arch=='resnet20': return ResNet20(classes)
    if arch=='femnist_cnn': return FEMNISTCNN(classes)
    if arch=='tabular_mlp':
        input_dim=int(cfg['model'].get('input_dim') or (bundle.input_shape[0] if bundle else 54)); return TabularMLP(input_dim,classes,tuple(cfg['model'].get('hidden_dims',[128,64])))
    if arch=='shakespeare_lstm':
        p=cfg['model']; return ShakespeareLSTM(classes,int(p.get('embedding_dim',64)),int(p.get('hidden_size',256)),int(p.get('num_layers',2)))
    raise ValueError(f'Unknown model architecture: {arch}')
