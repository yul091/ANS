import numpy as np
import torch
import os

import matplotlib.pyplot as plt

def distance(i, items, norm=2):
    target = items[i]
    if norm == 2:
        distance = np.sqrt(np.sum((items - target) ** 2, axis=1))
    else:
        distance = np.max(np.abs(items - target), axis=1)
    distance[i] = float("inf")
    return min(distance)

def distance_orginal(items_origin, items):
    distance = np.sqrt(np.sum((items_origin - items) ** 2, axis=1))
    return distance

def read_items(path):
    model = torch.load(os.path.join(path, "model.p"), map_location='cpu')
    items = model['embed_item.weight'].numpy()
    return items

# path1 = "/Users/aaronhe/Documents/CODE/ANS/kaggle/bpr/epoch0_auc0.5035_ndcg0.0459_hr0.1003_loss0.6931-2019-11-23-21-05-35"
# path2 = "/Users/aaronhe/Documents/CODE/ANS/kaggle/bpr/epoch22_auc0.7500_ndcg0.1998_hr0.3764_loss0.7695-2019-11-23-21-33-51"
# path1 = "/Users/aaronhe/Documents/CODE/ANS/amazon-movie-10-core/bpr/auc0.4995_ndcg0.0451_hr0.0994_loss0.6931-2019-11-21-16-31-34"
# path2 = "/Users/aaronhe/Documents/CODE/ANS/amazon-movie-10-core/bpr/auc0.8562_ndcg0.3718_hr0.6000_loss0.8078-2019-11-21-21-57-56"
# path1 = "/Users/aaronhe/Documents/CODE/ANS/ckpt/bpr/epoch0_auc0.4998_ndcg0.0445_hr0.0987_loss0.6931-2019-11-25-14-31-38"
# path2 = "/Users/aaronhe/Documents/CODE/ANS/ckpt/bpr/epoch46_auc0.8710_ndcg0.4351_hr0.6653_loss0.7278-2019-11-26-08-36-11"
path = "/Users/aaronhe/Documents/CODE/ANS/ckpt/mlp"
# path = "/Users/aaronhe/Documents/CODE/ANS/kaggle/ans/epoch133_auc0.7776_ndcg0.2016_hr0.3934_loss0.7519-2019-11-24-10-29-55"
# path = "/Users/aaronhe/Documents/CODE/ANS/amazon-movie-10-core/dns/auc0.8620_ndcg0.4071_hr0.6346_loss0.7126-2019-11-22-07-31-59"

items_origin = read_items(path)
items = read_items(path)
# dist = distance_orginal(items_origin, items)
dist = [distance(i, items, norm=2) for i in range(len(items))]
plt.plot(sorted(dist), color="black")
plt.title("Minimal L2 Distance Between Trained Items")
plt.grid()
plt.xlabel("Item ID")
plt.ylabel("L2 Distance of Embeddings")
plt.show()