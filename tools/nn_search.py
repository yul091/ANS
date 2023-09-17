import numpy as np
import faiss
from time import time

d = 16
nb = 8000
nt = 1500
nq = 1
np.random.seed(123)
std = 1
xb = np.random.normal(scale=std, size=(nb, d)).astype('float32')
xt = np.random.normal(scale=std, size=(nt, d)).astype('float32')
xq = np.zeros((nq, d)).astype('float32')

t1 = time()
index = faiss.index_factory(d, "IVF64,Flat")
index.train(xb)
index.add(xb)
# index.nprobe = 1000
t2 = time()
print(t2 - t1)
D, I = index.search(xq, 1000)
print(len(I[0]))
print(D[0])
t3 = time()
print(t3 - t2)

index.make_direct_map()
recons_before = np.vstack([index.reconstruct(i) for i in range(nb)])

# revert order of the 200 first vectors
nu = 200
index.update_vectors(np.arange(nu), xb[nu - 1::-1].copy())

recons_after = np.vstack([index.reconstruct(i) for i in range(nb)])

# make sure reconstructions remain the same
diff_recons = recons_before[:nu] - recons_after[nu - 1::-1]
assert np.abs(diff_recons).max() == 0

D2, I2 = index.search(xq, 5)

assert np.all(D == D2)

gt_map = np.arange(nb)
gt_map[:nu] = np.arange(nu, 0, -1) - 1
eqs = I.ravel() == gt_map[I2.ravel()]

assert np.all(eqs)