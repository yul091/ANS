import os
import pandas as pd

dataset = "../data/yelp"
train = pd.read_csv(os.path.join(dataset, "train.txt"), sep="\t", header=None)
test = pd.read_csv(os.path.join(dataset, "test.txt"), sep="\t", header=None)

# train consecutiveness
max_u = train[0].max()
max_i = train[1].max()
print(max_u, train[0].nunique() - 1)
print(max_i, train[1].nunique() - 1)

# test 
max_u_test = test[0].max()
max_i_test = test[1].max()
assert max_u_test <= max_u
assert max_i_test <= max_i

# 1:99
test = test.sort_values(by=[0, 2], ascending=False).reset_index(drop=True).to_numpy()
print(test[:10])
assert len(test) % 100 == 0
for k in range(len(test) // 100):
    batch = test[k*100: (k+1)*100]
    u = batch[0, 0]
    assert sum(batch[1:, 2]) == 0
    assert batch[0, 2] != 0
    for b in batch:
        assert b[0] == u


