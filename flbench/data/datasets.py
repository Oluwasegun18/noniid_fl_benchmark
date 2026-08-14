from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from .partitioning import load_or_create, iid_partition
from .types import FederatedDatasetBundle

CIFAR10_MEAN=(0.4914,0.4822,0.4465)
CIFAR10_STD=(0.2470,0.2435,0.2616)
CIFAR100_MEAN=(0.5071,0.4867,0.4408)
CIFAR100_STD=(0.2675,0.2565,0.2761)


def _image_transform(train: bool, augmentation: bool, mean, std):
    ops=[]
    if train and augmentation:
        ops += [transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(ops+[transforms.ToTensor(), transforms.Normalize(mean,std)])


def _split_indices(length: int, validation_fraction: float, seed: int, subset_size=None):
    rng=np.random.default_rng(seed)
    idx=np.arange(length); rng.shuffle(idx)
    if subset_size is not None:
        idx=idx[:min(int(subset_size),len(idx))]
    nval=max(1,int(len(idx)*validation_fraction))
    return idx[nval:].tolist(), idx[:nval].tolist()


def targets_of(dataset):
    if isinstance(dataset,Subset):
        base=targets_of(dataset.dataset)
        return np.asarray(base)[np.asarray(dataset.indices)]
    if isinstance(dataset,TensorDataset):
        return dataset.tensors[1].detach().cpu().numpy()
    if hasattr(dataset,'targets'):
        return np.asarray(dataset.targets)
    raise TypeError(f'Cannot extract class targets from {type(dataset).__name__}.')


def _artificial_partitions(train_dataset, cfg):
    part=cfg['partition']; method=str(part.get('method','dirichlet')).lower()
    n=int(part['num_clients'])
    if method == 'iid':
        return iid_partition(len(train_dataset), n, int(part.get('seed',1)))
    if method == 'dirichlet':
        return load_or_create(targets_of(train_dataset), part)
    raise ValueError(f'Unsupported artificial partition method: {method}')


def build_cifar10(cfg):
    data=cfg['data']; seed=int(cfg['experiment']['seed']); root=str(Path(data['data_dir']))
    train_aug=datasets.CIFAR10(root,train=True,download=True,transform=_image_transform(True,bool(data.get('augmentation',False)),CIFAR10_MEAN,CIFAR10_STD))
    train_eval=datasets.CIFAR10(root,train=True,download=True,transform=_image_transform(False,False,CIFAR10_MEAN,CIFAR10_STD))
    test=datasets.CIFAR10(root,train=False,download=True,transform=_image_transform(False,False,CIFAR10_MEAN,CIFAR10_STD))
    tr,va=_split_indices(len(train_eval),float(data.get('validation_fraction',0.1)),seed,data.get('subset_size'))
    train=Subset(train_aug,tr); validation=Subset(train_eval,va)
    return FederatedDatasetBundle(train,validation,test,_artificial_partitions(train,cfg),'classification',10,(3,32,32),metadata={'partition_source':'artificial'})


def build_cifar100(cfg):
    data=cfg['data']; seed=int(cfg['experiment']['seed']); root=str(Path(data['data_dir']))
    train_aug=datasets.CIFAR100(root,train=True,download=True,transform=_image_transform(True,bool(data.get('augmentation',False)),CIFAR100_MEAN,CIFAR100_STD))
    train_eval=datasets.CIFAR100(root,train=True,download=True,transform=_image_transform(False,False,CIFAR100_MEAN,CIFAR100_STD))
    test=datasets.CIFAR100(root,train=False,download=True,transform=_image_transform(False,False,CIFAR100_MEAN,CIFAR100_STD))
    tr,va=_split_indices(len(train_eval),float(data.get('validation_fraction',0.1)),seed,data.get('subset_size'))
    train=Subset(train_aug,tr); validation=Subset(train_eval,va)
    return FederatedDatasetBundle(train,validation,test,_artificial_partitions(train,cfg),'classification',100,(3,32,32),metadata={'partition_source':'artificial'})


def build_covtype(cfg):
    data=cfg['data']; seed=int(cfg['experiment']['seed'])
    bunch=fetch_covtype(data_home=str(Path(data['data_dir'])/'covtype'), download_if_missing=True)
    x=np.asarray(bunch.data,dtype=np.float32); y=np.asarray(bunch.target,dtype=np.int64)-1
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); rng.shuffle(idx)
    subset=data.get('subset_size')
    if subset is not None: idx=idx[:min(int(subset),len(idx))]
    x=x[idx]; y=y[idx]
    test_fraction=float(data.get('test_fraction',0.2)); val_fraction=float(data.get('validation_fraction',0.1))
    ntest=max(1,int(len(y)*test_fraction)); nval=max(1,int((len(y)-ntest)*val_fraction))
    test_idx=np.arange(ntest); val_idx=np.arange(ntest,ntest+nval); train_idx=np.arange(ntest+nval,len(y))
    scaler=StandardScaler().fit(x[train_idx])
    x_train=scaler.transform(x[train_idx]).astype(np.float32); y_train=y[train_idx]
    x_val=scaler.transform(x[val_idx]).astype(np.float32); y_val=y[val_idx]
    x_test=scaler.transform(x[test_idx]).astype(np.float32); y_test=y[test_idx]
    train=TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train))
    validation=TensorDataset(torch.from_numpy(x_val),torch.from_numpy(y_val))
    test=TensorDataset(torch.from_numpy(x_test),torch.from_numpy(y_test))
    return FederatedDatasetBundle(train,validation,test,_artificial_partitions(train,cfg),'classification',7,(x_train.shape[1],),metadata={'partition_source':'artificial','feature_scaling':'StandardScaler fit on train only'})


def _leaf_files(root: Path, split: str):
    candidates=[root/split, root/'data'/split, root/root.name/'data'/split]
    for folder in candidates:
        if folder.exists():
            files=sorted(folder.glob('*.json'))
            if files: return files
    return []


def _read_leaf_users(files: Iterable[Path]):
    out={}
    for file in files:
        payload=json.loads(file.read_text(encoding='utf-8'))
        users=payload.get('users', list(payload.get('user_data',{})))
        for user in users:
            out[user]=payload['user_data'][user]
    return out


def _natural_split_user_records(users: dict, validation_fraction: float, seed: int):
    rng=np.random.default_rng(seed); train_records=[]; val_records=[]; client_indices={}; client_names={}
    for cid,user in enumerate(sorted(users)):
        xs=list(users[user]['x']); ys=list(users[user]['y']); order=np.arange(len(ys)); rng.shuffle(order)
        nval=max(1,int(len(order)*validation_fraction)) if len(order)>1 else 0
        val_sel=set(order[:nval].tolist()); start=len(train_records)
        for j,(x,y) in enumerate(zip(xs,ys)):
            (val_records if j in val_sel else train_records).append((x,y))
        count=len(train_records)-start
        if count>0:
            client_indices[cid]=list(range(start,start+count)); client_names[cid]=user
    return train_records,val_records,client_indices,client_names


class FEMNISTRecords(Dataset):
    def __init__(self, records): self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        x,y=self.records[i]
        arr=np.asarray(x,dtype=np.float32).reshape(28,28)
        if arr.max()>1.0: arr=arr/255.0
        return torch.from_numpy(arr).unsqueeze(0), torch.tensor(int(y),dtype=torch.long)


def build_femnist(cfg):
    data=cfg['data']; root=Path(data['data_dir'])/'femnist'
    train_files=_leaf_files(root,'train'); test_files=_leaf_files(root,'test')
    if not train_files or not test_files:
        raise FileNotFoundError('FEMNIST expects LEAF-style JSON files under <data_dir>/femnist/{train,test}/ or <data_dir>/femnist/data/{train,test}/.')
    train_users=_read_leaf_users(train_files); test_users=_read_leaf_users(test_files)
    train_records,val_records,client_indices,client_names=_natural_split_user_records(train_users,float(data.get('validation_fraction',0.1)),int(cfg['experiment']['seed']))
    test_records=[(x,y) for user in sorted(test_users) for x,y in zip(test_users[user]['x'],test_users[user]['y'])]
    min_samples=int(cfg['partition'].get('min_samples_per_client',1))
    keep={cid:idx for cid,idx in client_indices.items() if len(idx)>=min_samples}
    if len(keep)!=len(client_indices):
        # Repack train dataset so client indices remain contiguous and valid.
        new_records=[]; new_indices={}; new_names={}
        for new_id,old_id in enumerate(sorted(keep)):
            start=len(new_records); new_records += [train_records[j] for j in keep[old_id]]
            new_indices[new_id]=list(range(start,len(new_records))); new_names[new_id]=client_names[old_id]
        train_records,client_indices,client_names=new_records,new_indices,new_names
    cfg['partition']['num_clients']=len(client_indices)
    return FederatedDatasetBundle(FEMNISTRecords(train_records),FEMNISTRecords(val_records),FEMNISTRecords(test_records),client_indices,'classification',62,(1,28,28),client_names=client_names,metadata={'partition_source':'natural_writer'})


DEFAULT_SHAKESPEARE_VOCAB=list("\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}")

class ShakespeareRecords(Dataset):
    def __init__(self, records, vocab, sequence_length=80):
        self.records=records; self.vocab=vocab; self.lookup={c:i for i,c in enumerate(vocab)}; self.sequence_length=int(sequence_length); self.pad=0
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        x,y=self.records[i]; text=str(x)
        ids=[self.lookup.get(ch,self.pad) for ch in text[-self.sequence_length:]]
        if len(ids)<self.sequence_length: ids=[self.pad]*(self.sequence_length-len(ids))+ids
        target=self.lookup.get(str(y)[0] if str(y) else '\n',self.pad)
        return torch.tensor(ids,dtype=torch.long), torch.tensor(target,dtype=torch.long)


def build_shakespeare(cfg):
    data=cfg['data']; root=Path(data['data_dir'])/'shakespeare'
    train_files=_leaf_files(root,'train'); test_files=_leaf_files(root,'test')
    if not train_files or not test_files:
        raise FileNotFoundError('Shakespeare expects LEAF-style JSON files under <data_dir>/shakespeare/{train,test}/ or <data_dir>/shakespeare/data/{train,test}/.')
    train_users=_read_leaf_users(train_files); test_users=_read_leaf_users(test_files)
    train_records,val_records,client_indices,client_names=_natural_split_user_records(train_users,float(data.get('validation_fraction',0.1)),int(cfg['experiment']['seed']))
    test_records=[(x,y) for user in sorted(test_users) for x,y in zip(test_users[user]['x'],test_users[user]['y'])]
    vocab=list(data.get('vocabulary',DEFAULT_SHAKESPEARE_VOCAB)); seq_len=int(data.get('sequence_length',80))
    min_samples=int(cfg['partition'].get('min_samples_per_client',1))
    keep={cid:idx for cid,idx in client_indices.items() if len(idx)>=min_samples}
    if len(keep)!=len(client_indices):
        new_records=[]; new_indices={}; new_names={}
        for new_id,old_id in enumerate(sorted(keep)):
            start=len(new_records); new_records += [train_records[j] for j in keep[old_id]]
            new_indices[new_id]=list(range(start,len(new_records))); new_names[new_id]=client_names[old_id]
        train_records,client_indices,client_names=new_records,new_indices,new_names
    cfg['partition']['num_clients']=len(client_indices)
    return FederatedDatasetBundle(ShakespeareRecords(train_records,vocab,seq_len),ShakespeareRecords(val_records,vocab,seq_len),ShakespeareRecords(test_records,vocab,seq_len),client_indices,'next_character',len(vocab),(seq_len,),vocabulary=vocab,client_names=client_names,metadata={'partition_source':'natural_speaking_role','sequence_length':seq_len})


DATASET_REGISTRY={
    'cifar10': build_cifar10,
    'cifar100': build_cifar100,
    'covtype': build_covtype,
    'femnist': build_femnist,
    'shakespeare': build_shakespeare,
}


def build_dataset(cfg):
    name=str(cfg['data']['dataset']).lower()
    if name not in DATASET_REGISTRY: raise ValueError(f'Unsupported dataset: {name}')
    bundle=DATASET_REGISTRY[name](cfg)
    cfg['model']['num_classes']=bundle.num_classes
    return bundle

# Backward compatibility.
def load_cifar10(data_dir, validation_fraction, seed, augmentation=False, subset_size=None):
    cfg={'experiment':{'seed':seed},'data':{'dataset':'cifar10','data_dir':data_dir,'validation_fraction':validation_fraction,'augmentation':augmentation,'subset_size':subset_size},'partition':{'method':'iid','num_clients':1,'seed':seed,'cache_dir':'partition_cache','min_samples_per_client':1}}
    b=build_cifar10(cfg); return b.train_dataset,b.validation_dataset,b.test_dataset
