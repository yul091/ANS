import numpy as np
from time import time
import bisect

def improved(prob_matrix, items):
    # transpose here for better data locality later
    cdf = np.cumsum(prob_matrix.T, axis=1)
    # random numbers are expensive, so we'll get all of them at once
    ridx = np.random.random(size=n)
    # the one loop we can't avoid, made as simple as possible
    idx = np.zeros(n, dtype=int)
    for i, r in enumerate(ridx):
        idx[i] = np.searchsorted(cdf[i], r)
    # fancy indexing all at once is faster than indexing in a loop
    return items[idx]


def original(prob_matrix, items):
    choices = np.zeros((n,))
    # This is slow, because of the loop in Python
    for i in range(n):
        choices[i] = np.random.choice(items, p=prob_matrix[:,i])
    return choices

def bisec(prob_matrix, items):
    # transpose here for better data locality later
    cdf = np.cumsum(prob_matrix.T, axis=1)
    # the one loop we can't avoid, made as simple as possible
    idx = np.zeros(n, dtype=int)
    for i in range(n):
        cs = cdf[i]
        idx[i] = bisect.bisect(cs, np.random.random() * cs[-1])
    # fancy indexing all at once is faster than indexing in a loop
    return items[idx]

def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]

m = 25815
n = 256

items = np.arange(m)
prob_weights = np.random.rand(m,n)
prob_matrix = prob_weights / sum(prob_weights)

t1 = time()
for _ in range(1000):
    bisec(prob_matrix, items)
print(time()-t1)

t1 = time()
for _ in range(1000):
    original(prob_matrix, items)
print(time()-t1)

t1 = time()
for _ in range(1000):
    improved(prob_matrix, items)
print(time()-t1)

t1 = time()
for _ in range(1000):
    vectorized(prob_matrix, items)
print(time()-t1)