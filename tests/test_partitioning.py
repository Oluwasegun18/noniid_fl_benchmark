import numpy as np
from flbench.data.partitioning import dirichlet_partition
def test_reproducibility():
    y=np.repeat(np.arange(10),100); a=dirichlet_partition(y,5,0.5,7,20); b=dirichlet_partition(y,5,0.5,7,20); assert a==b
def test_all_samples_used():
    y=np.repeat(np.arange(10),100); p=dirichlet_partition(y,5,0.5,7,20); assert sorted(i for v in p.values() for i in v)==list(range(len(y)))
