import numpy as np
import time
from collections import Counter

def choice(options, probs):
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]

options = ['a','b','c','d']
probs = [0.2,0.6,0.15,0.05]
runs = 100000

now = time.time()
temp = []
for i in range(runs):
    op = choice(options,probs)
    temp.append(op)
temp = Counter(temp)
for op,x in temp.items():
    print(op,x/runs)
print(time.time()-now)

print("")
now = time.time()
temp = []
for i in range(runs):
    op = np.random.choice(options,p = probs)
    temp.append(op)
temp = Counter(temp)
for op,x in temp.items():
    print(op,x/runs)
print(time.time()-now)