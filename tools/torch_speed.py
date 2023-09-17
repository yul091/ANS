import torch
from time import time

N = 1000

x = torch.Tensor(16)

t1 = time()
for i in range(N):
    y = str(x.tolist())
print(time() - t1)

t2 = time()
for i in range(N):
    y = tuple(x.tolist())
print(time() - t2)