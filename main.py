import sys
import os
import random

from mf.model import MF, GMF, VMF, MLP, NCF, ConvNCF
from mf.metrics import metrics

from mf.dataset import Dataset, BPRDataset, VisualDataset, VisualBPRDataset, DNSDataset, ANSDataset, ANS2Dataset, FastDataset, GANSDataset, PDNSDataset, FANSDataset, PANSDataset, TANSDataset, POPDataset
from mf.utils import DataSpliter
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

from time import time
from tqdm import tqdm

import argparse

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run ANS.")
    # data
    parser.add_argument('--data', type=str, default='ml-1m',
                        help='Input data path.')
    # model
    parser.add_argument('--strategy', type=str, default='mf-ans',
                        help='Choose training model from mf, gmf, mlp, ncf and strategy from pointwise, bpr, dns, ans')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='Hidden dimension of embedding')
    parser.add_argument('--biased', type=bool, default=False,
                        help='Whether use bias terms in model')
    parser.add_argument('--choice', type=str, default="soft",
                        help='Choose choice strategy from hard, soft and dynamic')
    parser.add_argument('--layers', type=eval, default="[64,32,16,8]",
                        help='Layers for MLP model')
    parser.add_argument('--k', type=int, default=10,
                        help='Negative sampling size')
    parser.add_argument('--load_ckpt', type=str, default=None,
                        help='Checkpoint loading path')
    parser.add_argument('--load_gmf_ckpt', type=str, default=None,
                        help='Checkpoint gmf loading path')
    parser.add_argument('--load_mlp_ckpt', type=str, default=None,
                        help='Checkpoint mlp loading path')
    parser.add_argument('--bp', action="store_true", default=False,
                        help='Whether backprop for randomly choosing negative item in pair-wise learning')
    # training
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--sampling', type=int, default=3,
                        help='For pointwise training, X negative samples for each positive item.')
    parser.add_argument('--l2', type=str, default="0.05",
                        help='L2 regularization for embedding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Choose running device between cpu and cuda')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether shuffle training set for every epoch')
    parser.add_argument('--optim', type=str, default="SGD",
                        help='Choose optimizer from SGD, Adagrad and Adam')
    parser.add_argument('--eps', type=float, default=1,
                        help='Radius for adversarial negative sampling')
    parser.add_argument('--ks', type=str, default='[5,10,20,50]',
                        help='NDCG@k and HR@k evaluation')
    parser.add_argument('--eval_workers', type=int, default=8,
                        help='Number of workers to speed up evaluation')

    return parser.parse_args()

def main():

    args = parse_args()

    if os.path.exists("data/%s.txt" % args.data):
        data = pd.read_csv("data/%s.txt" % args.data, header=None, sep="\t")    
    else:
        data = pd.read_csv("data/%s/train.txt" % args.data, header=None, sep="\t")    

    args.num_items = data[1].max()
    args.num_users = data[0].max()
    args.loss = nn.BCEWithLogitsLoss()
    args.threshold = 1
    args.metrics = ["ndcg", "hr", "auc","loss"]
    args.adv = 1e-4
    args.save_dir = "ckpt/" + args.data + "/" + args.strategy
    args.dataset = "data/" + args.data
    args.l2 = eval(str(args.l2)) if "{" in args.l2 else float(args.l2)
    args.ks = eval(args.ks)

    if args.optim == "SGD":
        args.optim = optim.SGD
    elif args.optim == "Adam":
        args.optim = optim.Adam
    elif args.optim == "Adagrad":
        args.optim = optim.Adagrad

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
    if "gmf" in args.strategy:
        model = GMF(args)
    elif "mlp" in args.strategy:
        model = MLP(args)
    elif "mf" in args.strategy:
        model = MF(args)
    elif "conv" in args.strategy:
        model = ConvNCF(args)
    elif "ncf" in args.strategy:
        model = NCF(args)

    if args.device == "cuda":
        model.cuda()
        args.loss.cuda()

    # model_dict = model.state_dict()
    # print(model_dict.keys())

    if args.load_ckpt is not None or args.load_mlp_ckpt is not None:
        
        model_dict = model.state_dict()
        #print(model_dict)
        if "ncf" in args.strategy:
            gmf_model, mlp_model = torch.load(args.load_gmf_ckpt), torch.load(args.load_mlp_ckpt)
            mlp_dict = {k: v for k, v in mlp_model.items() if k in model_dict and "mlp" in k}
            model_dict.update(mlp_dict)
            model_dict['embed_user.weight'] = torch.cat((gmf_model['embed_user.weight'], mlp_model['embed_user.weight']), dim=1)
            model_dict['embed_item.weight'] = torch.cat((gmf_model['embed_item.weight'], mlp_model['embed_item.weight']), dim=1)
            model_dict['fc.weight'] = torch.cat((gmf_model['fc.weight']*0.5, mlp_model['fc.weight']*0.5), dim=1)
            print(gmf_model['embed_user.weight'])
        else:
            pretrained_dict = torch.load(args.load_ckpt)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 4. print
            print("loading parameters from", pretrained_dict.keys())
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print(model.embed_user.weight)

    print(args)
    print(model)

    if "bpr" in args.strategy:
        model.fit(train=BPRDataset(train), valid=FastDataset(test), args=args, type="bpr")

    if "pop" in args.strategy:
        model.fit(train=POPDataset(train), valid=FastDataset(test), args=args, type="bpr")

    elif "pdns" in args.strategy:
        model.fit(train=PDNSDataset(train), valid=FastDataset(test), args=args, type="pdns")

    elif "pans" in args.strategy:
        model.fit(train=PANSDataset(train), valid=FastDataset(test), args=args, type="pdns")

    elif "tans" in args.strategy:
        model.fit(train=TANSDataset(train), valid=FastDataset(test), args=args, type="tans")

    elif "fans" in args.strategy:
        model.fit(train=FANSDataset(train), valid=FastDataset(test), args=args, type="fans")

    elif "dns" in args.strategy:
        model.fit(train=DNSDataset(train), valid=FastDataset(test), args=args, type="dns")
    
    elif "gans" in args.strategy:
        model.fit(train=GANSDataset(train), valid=FastDataset(test), args=args, type="gans")

    elif "ans" in args.strategy:
        model.fit(train=ANSDataset(train), valid=FastDataset(test), args=args, type="dns")

    else:    
        model.fit(train=Dataset(train), valid=FastDataset(test), args=args, type="implicit")

if __name__ == "__main__":
    setup_seed(5583)
    main()
