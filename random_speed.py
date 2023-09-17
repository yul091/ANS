import random
import numpy as np
from time import time

ITEMS = 10000
K = 10

a = set(range(ITEMS))
b = [set([np.random.randint(ITEMS) for i in range(np.random.randint(200))]) for _ in range(100)]

t1 = time()
for c in b:
    usernotread = a-c
    items = random.sample(a, K)
print(time()-t1)

a = list(a)
t2 = time()
for c in b:
    p = np.ones(ITEMS)
    p[list(c)] = 0
    items = np.random.choice(a, K, p=p/p.sum())
print(time()-t2)