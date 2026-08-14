from pathlib import Path
import hashlib, json
import numpy as np

def dirichlet_partition(targets, num_clients, alpha, seed, min_samples=1, attempts=500):
    rng=np.random.default_rng(seed); classes=np.unique(targets)
    for _ in range(attempts):
        out={i:[] for i in range(num_clients)}
        for cls in classes:
            idx=np.where(targets==cls)[0]; rng.shuffle(idx); counts=rng.multinomial(len(idx),rng.dirichlet(np.full(num_clients,alpha)))
            s=0
            for i,c in enumerate(counts): out[i].extend(idx[s:s+c].tolist()); s+=c
        if min(map(len,out.values()))>=min_samples:
            for v in out.values(): rng.shuffle(v)
            return out
    raise RuntimeError('Unable to satisfy minimum client size; reduce min_samples or increase alpha.')

def load_or_create(targets,cfg):
    key=f"cifar10|{cfg['alpha']}|{cfg['num_clients']}|{cfg['seed']}|{len(targets)}"
    pid=hashlib.sha1(key.encode()).hexdigest()[:16]; path=Path(cfg['cache_dir'])/f'{pid}.json'; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raw=json.loads(path.read_text()); return {int(k):v for k,v in raw['client_indices'].items()}
    p=dirichlet_partition(targets,int(cfg['num_clients']),float(cfg['alpha']),int(cfg['seed']),int(cfg['min_samples_per_client']))
    path.write_text(json.dumps({'partition_id':pid,'client_indices':p,'client_sizes':{k:len(v) for k,v in p.items()}},indent=2))
    return p



def iid_partition(num_samples: int, num_clients: int, seed: int):
    rng=np.random.default_rng(seed); indices=np.arange(num_samples); rng.shuffle(indices)
    chunks=np.array_split(indices,num_clients)
    return {i:chunk.astype(int).tolist() for i,chunk in enumerate(chunks)}
