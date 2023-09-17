import sys
import os
import random

from mf.model import MF, GMF, VMF, MLP, NCF
from mf.metrics import metrics
from mf.dataset import Dataset, BPRDataset, VisualDataset, VisualBPRDataset, DNSDataset, ANSDataset
from mf.utils import DataSpliter
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

from time import time
from tqdm import tqdm

from matplotlib import pyplot as plt

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def create_adv(loss, model, candiate):
    loss.backward()
    return model.embed_item.weight.grad[candiate]

def test_linear(user, item, model, device, eps=[0.1, 0.5, 1, 2, 3, 5, 10], k=None, opt=None):
    user = torch.LongTensor(user)
    item = torch.LongTensor(item)

    if device == "cuda":
        user, item = user.cuda(), item.cuda()
    r = model.forward(user, item)

    adv = create_adv(-(1-r.sigmoid()).log(), model, item)
    r = [r.item()]
    opt.zero_grad()
    r.extend([model.forward(user, item, e * (adv/adv.norm())).item() for e in eps])

    return r

def test_bpr(data, strategy="bpr"):
    class Args(object):
        pass

    args = Args()
    args.sampling = 4
    args.hidden_dim = 16
    args.num_items = data[1].max()
    args.num_users = data[0].max()
    args.loss = nn.BCEWithLogitsLoss()
    args.optim = optim.SGD
    args.epochs = 1000
    args.lr = .001
    args.batch_size = 32
    args.threshold = 1
    args.l2 = 0
    args.device = "cuda"
    args.shuffle = True
    args.biased = False
    args.metrics = [ "ndcg", "hr", "auc","loss"]
    args.k = 4
    args.save_dir = "ckpt/kaggle/" + strategy
    args.dataset = "data/kaggle"
    args.layers = [64, 32, 16, 8]

    if os.path.exists(args.dataset):
        print("loading dataset from %s ..." % args.dataset)
        train = pd.read_csv(os.path.join(args.dataset, "train.txt"), header=None, sep="\t")
        test = pd.read_csv(os.path.join(args.dataset, "test.txt"), header=None, sep="\t")

    else:
        print("generating dataset to %s ..." % args.dataset)
        train, test = DataSpliter.leave_k_out_split(data)
        args.global_mean = train[2].mean()

        neg_sample = 99
        neg_pairs = []
        users = set([])
        dataset_temp = Dataset(data)
        for user in list(test[0].unique()):
            items = dataset_temp.interact_status[user]
            users.add(user)
            neg_pairs.extend([[user, i, 0, 0] for i in \
                            random.sample(dataset_temp.item_pool-items, neg_sample)])
        neg_valid = pd.DataFrame(data=neg_pairs)
        test = test.append(neg_valid).reset_index(drop=True)

        os.makedirs(args.dataset)
        train.to_csv(os.path.join(args.dataset, "train.txt"), header=None, sep="\t", index=None)
        test.to_csv(os.path.join(args.dataset, "test.txt"), header=None, sep="\t", index=None)

    # model initialization
    model = NCF(args)

    if args.device == "cuda":
        model.cuda()
        args.loss.cuda()

    # model.load_state_dict(torch.load("/home/zhankui/ANS/ckpt/ml-100k/mf/epoch12_ndcg0.3438_hr0.6045_auc0.8746_loss0.1408-2019-11-30-23-34-42/model.p"))
    model.load_state_dict(torch.load("/home/zhankui/ANS/ckpt/kaggle/bpr/epoch7_ndcg0.2256_hr0.4150_auc0.7821_loss0.1350-2019-12-01-15-24-25/model.p"))
    # model.load_state_dict(torch.load("/home/zhankui/ANS/ckpt/ml-100k/bpr/epoch838_auc0.8704_ndcg0.3312_hr0.5854_loss0.9521-2019-11-23-09-12-30/model.p"))
    model.evaluate(Dataset(test), args)

    opt = args.optim(model.parameters(), lr=args.lr)
    header = [-1, 0] + list(np.arange(0.1, 10, 0.1))
    obs = []

    for i, row in tqdm(test.iterrows()):
        obs.append([row[3]] + test_linear([row[0]], [row[1]], model, args.device, eps = np.arange(0.1, 10, 0.1), opt=opt))
    
    pd.DataFrame(obs).to_csv("observation-ncf-kaggle.csv", index=None, header=header)

def visualization(path):
    df = pd.read_csv(path)
    x = [round(float(i), 2) for i in df.columns[1:20]]
    for i in random.sample(range(len(df)), 1000):
        p = np.array(df.iloc[i].tolist()[1:20])
        p -= min(p)
        p /= max(p)
        plt.plot(x, p, alpha=0.01, color="black")
    plt.title("Predicted Score Changing toward Gradient Direction")
    plt.xlabel("Distance toward Gradient Direction")
    plt.ylabel("(Normalized) Predicted Score Changing")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    setup_seed(5583)
    data = pd.read_csv("data/kaggle.txt", header=None, sep="\t")    
    test_bpr(data, sys.argv[1])
    # visualization("observation-ncf-1000-kaggle.csv", int(sys.argv[1]))
    # visualization("obs/observation-ncf-1000-kaggle.csv")
