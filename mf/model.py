import os
import torch
import pandas as pd
import numpy as np
import collections 

import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import time
import faiss

from metrics import metrics, metrics_fast
from utils import AvgMeter, Grid

from multiprocessing import Pool, cpu_count

class MF(torch.nn.Module):
    """ Matrix Factorization Model for ReAL quick experiment
        Follow the idea of y' = pq
    """
    def __init__(self, args):
        super(MF, self).__init__()
        # store args
        self.args = args
        self.epoch = 0

        # initialize the parameters
        self.num_users = getattr(args, "num_users")
        self.num_items = getattr(args, "num_items")
        self.hidden_dim = getattr(args, "hidden_dim")

        # initialize item and user embedding
        # "+1" for some scnario that using padding index, or starting from 1
        self.embed_user = torch.nn.Embedding(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.hidden_dim
        )
        self.embed_item = torch.nn.Embedding(
            num_embeddings=self.num_items+1, 
            embedding_dim=self.hidden_dim
        )

        if args.biased:

            self.bias_user = torch.nn.Embedding(
                num_embeddings=self.num_users+1, 
                embedding_dim=1
            )
            self.bias_item = torch.nn.Embedding(
                num_embeddings=self.num_items+1, 
                embedding_dim=1
            )

            self.bias = nn.Parameter(torch.tensor(0.))

        self.init()

    def init(self):

        # !!Embedding initialization below is very important!!
        # N(0,1) ruins our model for rating, use N(0,.1) instead
        self.embed_user.weight.data.normal_(0,.01)
        self.embed_item.weight.data.normal_(0,.01)

        try:
            self.bias_user.weight.data.fill_(0)
            self.bias_item.weight.data.fill_(0)
            self.bias = nn.Parameter(torch.tensor(self.args.global_mean))
        except:
            pass

    def forward(self, user_indices, item_indices, adv=None):
        # user embeddings
        self.p_u = self.embed_user(user_indices)
        # item embeddings
        self.q_i = self.embed_item(item_indices) 
        # adv training
        self.q_i = self.q_i + adv if adv is not None else self.q_i
        
        if self.args.biased:
            # user bias
            self.b_u = self.bias_user(user_indices).squeeze()
            # item bias
            self.b_i = self.bias_item(item_indices).squeeze()
            # model
            return (self.p_u * self.q_i).sum(-1) + self.b_u + self.b_i + self.bias
        
        return (self.p_u * self.q_i).sum(-1)

    def reg_loss(self, l2):
        reg_u = 0
        reg_i = 0
        if type(l2) != dict:
            if l2 == 0: return 0, 0
            if self.args.biased:
                return l2 * ((self.p_u**2).sum() + (self.b_u**2).sum()), \
                    l2 * ((self.p_u**2).sum() + (self.b_i**2).sum())
            else:
                return l2 * (self.p_u**2).sum(), l2 * (self.q_i**2).sum()

        for n, i in l2.items():
            if "p_u" in n: reg_u += i * (self.p_u**2).sum()
            if "q_i" in n: reg_i += i * (self.q_i**2).sum()
            if "b_u" in n: reg_u += i * (self.b_u**2).sum()
            if "b_i" in n: reg_i += i * (self.b_i**2).sum()
        return reg_u, reg_i

    def fit(self, train, valid=None, args=None, type="explicit"):
        if type == "explicit":
            return self._explicit(train, valid, args)
        elif type == "implicit":
            return self._implicit(train, valid, args)
        elif type == "bpr":
            return self._bpr(train, valid, args)
        elif type == "dns":
            return self._dns(train, valid, args)
        elif type == "ans":
            return self._ans(train, valid, args)
        elif type == "gans":
            return self._gans(train, valid, args)
        elif type == "pdns":
            return self._pdns(train, valid, args)
        elif type == "fans":
            return self._fans(train, valid, args)
        elif type == "tans":
            return self._tans(train, valid, args)
        else: 
            raise ValueError
    
    def _explicit(self, train, valid, args):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr, weight_decay=args.l2)
        # loss
        criterion = args.loss
        # training
        self.train()
        for epoch in range(args.epochs):
            self.epoch = epoch
            dataloader = train.build(args.sampling, batch_size=args.batch_size, device=args.device, shuffle=args.shuffle)
            for batch in tqdm(dataloader, desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                # forward
                r = batch[-1]
                r_ = self.forward(*batch[:-1])
                reg_u, reg_i = self.reg_loss(args.l2)
                # loss
                loss = criterion(r_, r) + reg_u + reg_i
                # gradients
                loss.backward()
                # update
                optimizer.step()

            print("training loss:", loss.item())            
            # validation
            if valid is not None:
                self.evaluate(valid, args)

    def _implicit(self, train, valid, args):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        # pos loss
        pos_loss_meter = AvgMeter()
        # neg loss
        neg_loss_meter = AvgMeter()
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            dataloader = train.build(args.sampling, batch_size=args.batch_size, threshold=args.threshold, device=args.device, shuffle=args.shuffle)
            for batch in tqdm(dataloader, desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                # forward
                r = batch[-1]
                r_ = self.forward(*batch[:-1])
                reg_u, reg_i = self.reg_loss(args.l2)
                # loss
                loss = criterion(r_, r) + reg_u + reg_i
                # gradients
                loss.backward()
                # update
                optimizer.step()
                loss_meter.update(loss.item())
                # pos loss
                pos_idx = r.nonzero()
                pos_loss_meter.update(criterion(r_[pos_idx], r[pos_idx]).item())
                # neg loss
                neg_idx = (r == 0).nonzero()
                neg_loss_meter.update(criterion(r_[neg_idx], r[neg_idx]).item())
            print("training loss:", loss_meter.avg)
            print("neg training loss:", neg_loss_meter.avg)
            print("pos training loss:", pos_loss_meter.avg)
            loss_meter.reset()
            neg_loss_meter.reset()
            pos_loss_meter.reset()

    def _bpr(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        loss_meter = AvgMeter()
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            dataloader = train.build(1, batch_size=args.batch_size, device=args.device, shuffle=args.shuffle)
            for u, i, j in tqdm(dataloader, desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_j = self.forward(u, j)
                reg_u, reg_j = self.reg_loss(args.l2)
                # log loss
                loss = - (r_i - r_j).sigmoid().log().sum() + reg_u + reg_i + reg_j
                # gradients
                loss.backward()
                # update
                optimizer.step()
                loss_meter.update(loss.item())
            # validation
            print("training loss:", loss_meter.avg)
            loss_meter.reset()

    def _dns(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        neg_quality = AvgMeter()
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(batch_size=args.batch_size, device=args.device, shuffle=args.shuffle, k=args.k)
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, j = train.next_batch(b, model=self, k=args.k, eps=args.eps, opt=optimizer, choice=args.choice, bp=args.bp, l2=args.l2)
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_j = self.forward(u, j)
                reg_u, reg_j = self.reg_loss(args.l2)
                # log loss
                loss = - (r_i - r_j).sigmoid().log().sum() + reg_u + reg_i + reg_j
                # gradients
                loss.backward()
                # update
                optimizer.step()
                loss_meter.update(loss.item())
                    
            # validation
            print("training loss:", loss_meter.avg)
            print("epoch negative quality:", neg_quality.avg)

            loss_meter.reset()
            neg_quality.reset()

    def _gans(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        neg_quality = AvgMeter()
        num, item = self.embed_item.weight.size()
        grid = Grid(args.eps, item, num, self.embed_item.weight)
        # TODO: why epoch1 is so lower
        # grid = None
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(batch_size=args.batch_size, device=args.device, shuffle=args.shuffle, k=args.k)
            length = []
            rj_list, r_j_list = [], []
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, n, j, l = train.next_batch(b, model=self, k=args.k, eps=args.eps, opt=optimizer, choice=args.choice, grid=grid)
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_n = self.forward(u, n)
                r_j = self.forward(u, j)
                reg_u, reg_j = self.reg_loss(args.l2)
                rj_list.extend(r_n.tolist())
                r_j_list.extend(r_j.tolist())
                # log loss
                loss = - (r_i - r_j).sigmoid().log().sum() + reg_u + reg_i + reg_j
                # gradients
                loss.backward()
                # update weight
                optimizer.step()
                loss_meter.update(loss.item())
                # update grid
                grid.update_batch(self.embed_item.weight, list(i.cpu().numpy()) + list(j.cpu().numpy()))
                length.extend(l)
                print("mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(rj_list), np.var(rj_list), np.max(rj_list), np.min(rj_list)))
                print("mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(r_j_list), np.var(r_j_list), np.max(r_j_list), np.min(r_j_list)))

            print("mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(length), np.var(length), np.max(length), np.min(length)))

            # validation
            print("training loss:", loss_meter.avg)
            print("epoch negative quality:", neg_quality.avg)

            loss_meter.reset()
            neg_quality.reset()

    def _fans(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        for epoch in range(args.epochs):
            # faiss
            self.index = faiss.index_factory(self.hidden_dim, "IVF64,Flat")
            embedding = self.embed_item.weight.detach().cpu().numpy().astype("float32")
            self.index.train(embedding)
            self.index.add(embedding)
            self.index.make_direct_map()
            self.index.nprobe = 10
            # epoch
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(batch_size=args.batch_size, device=args.device, shuffle=args.shuffle, k=args.k)
            length = []
            rj_list, r_j_dns_list, r_j_list = [], [], []
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, j, j_dns, j_, l = train.next_batch(b, model=self, k=args.k, eps=args.eps, opt=optimizer, choice=args.choice, grid=self.index)
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_j = self.forward(u, j)
                reg_u, reg_j = self.reg_loss(args.l2)
                r_j_ = self.forward(u, j_)
                reg_u, reg_j_ = self.reg_loss(args.l2)
                r_j_dns = self.forward(u, j_dns)
                reg_u, reg_j_dns = self.reg_loss(args.l2)
                rj_list.extend(r_j.tolist())
                r_j_dns_list.extend(r_j_dns.tolist())
                r_j_list.extend(r_j_.tolist())
                # log loss
                # loss = - (r_i - r_j).sigmoid().log().sum() - 0.1*(r_i - r_j_).sigmoid().log().sum() + reg_u + reg_i + reg_j + reg_j_
                loss = - (r_i - r_j).sigmoid().log().sum() - (r_i - r_j_).sigmoid().log().sum() + reg_j + reg_u + reg_i + reg_j_
                # loss = - (r_i - r_j_dns).sigmoid().log().sum() + reg_u + reg_i + reg_j_dns
                # loss = - (r_i - r_j).sigmoid().log().sum() + reg_u + reg_i + reg_j
                # gradients
                loss.backward()
                # update weight
                optimizer.step()
                loss_meter.update(loss.item())
                # update faiss
                self.index.update_vectors(i.cpu().numpy(), self.embed_item.weight[i].detach().cpu().numpy().astype("float32"))
                self.index.update_vectors(j.cpu().numpy(), self.embed_item.weight[j].detach().cpu().numpy().astype("float32"))
                self.index.update_vectors(j_.cpu().numpy(), self.embed_item.weight[j_].detach().cpu().numpy().astype("float32"))
                length.extend(l)

            print("Uni: mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(rj_list), np.var(rj_list), np.max(rj_list), np.min(rj_list)))
            print("DNS: mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(r_j_dns_list), np.var(r_j_dns_list), np.max(r_j_dns_list), np.min(r_j_dns_list)))    
            print("ANS: mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(r_j_list), np.var(r_j_list), np.max(r_j_list), np.min(r_j_list)))    

            print("K: mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(length), np.var(length), np.max(length), np.min(length)))

            # validation
            print("training loss:", loss_meter.avg)

            loss_meter.reset()

    def _tans(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        for epoch in range(args.epochs):
            # epoch
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(batch_size=args.batch_size, device=args.device, shuffle=args.shuffle, k=args.k)
            length = []
            rj_list, r_j_list = [], []
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, j, j_, l = train.next_batch(b, model=self, k=args.k, eps=args.eps, opt=optimizer, choice=args.choice, grid=None)
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_j = self.forward(u, j)
                reg_u, reg_j = self.reg_loss(args.l2)
                r_j_ = self.forward(u, j_)
                reg_u, reg_j_ = self.reg_loss(args.l2)
                rj_list.extend(r_j.tolist())
                r_j_list.extend(r_j_.tolist())
                # log loss
                # loss = - (r_i - r_j).sigmoid().log().sum() - (r_i - r_j_).sigmoid().log().sum() + reg_u + reg_i + reg_j + reg_j_
                loss = - (r_i - r_j_).sigmoid().log().sum() + reg_u + reg_i + reg_j_
                # gradients
                loss.backward()
                # update weight
                optimizer.step()
                loss_meter.update(loss.item())
                length.extend(l)

            print("mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(rj_list), np.var(rj_list), np.max(rj_list), np.min(rj_list)))
            print("mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(r_j_list), np.var(r_j_list), np.max(r_j_list), np.min(r_j_list)))    

            print("mean: %.2f, var: %.2f, max: %.2f, min: %.2f" % (np.mean(length), np.var(length), np.max(length), np.min(length)))

            # validation
            print("training loss:", loss_meter.avg)

            loss_meter.reset()

    def _pdns(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        # pos loss
        pos_loss_meter = AvgMeter()
        # neg loss
        neg_loss_meter = AvgMeter()
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(n_neg=args.sampling, batch_size=args.batch_size, device=args.device, shuffle=args.shuffle)
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, r = train.next_batch(batch=b, model=self, k=args.k, loss=criterion, eps=args.eps, opt=optimizer, choice=args.choice)
                # forward
                reg_u_neg, reg_i_neg = self.reg_loss(args.l2) if "pans" in args.strategy else (0, 0)
                r_ = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                # loss
                loss = criterion(r_, r) + reg_u + reg_i + reg_u_neg + reg_i_neg
                # gradients
                loss.backward()
                # update weight
                optimizer.step()
                loss_meter.update(loss.item())
                # pos loss
                pos_idx = r.nonzero()
                pos_loss_meter.update(criterion(r_[pos_idx], r[pos_idx]).item())
                # neg loss
                neg_idx = (r == 0).nonzero()
                neg_loss_meter.update(criterion(r_[neg_idx], r[neg_idx]).item())
            print("training loss:", loss_meter.avg)
            print("neg training loss:", neg_loss_meter.avg)
            print("pos training loss:", pos_loss_meter.avg)
            loss_meter.reset()
            neg_loss_meter.reset()
            pos_loss_meter.reset()

    def _ans(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        neg_quality = AvgMeter()
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(batch_size=args.batch_size, device=args.device, shuffle=args.shuffle)
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, j, j_ = train.next_batch(b, model=self, k=args.k, opt=optimizer, choice=args.choice)
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_j = self.forward(u, j)
                r_j_ = self.forward(u, j_)
                reg_u, reg_j = self.reg_loss(args.l2)

    def _gans(self, train, valid=None, args=None):
        # optimizer
        optimizer = args.optim(self.parameters(), lr=args.lr)
        # loss
        criterion = args.loss
        loss_meter = AvgMeter()
        neg_quality = AvgMeter()
        num, item = self.embed_item.weight.size()
        grid = Grid(args.eps, item, num, self.embed_item.weight)
        for epoch in range(args.epochs):
            self.epoch = epoch
            # validation
            if valid is not None:
                self.evaluate(valid, args)
            # training
            self.train()
            n_batch = train.build(batch_size=args.batch_size, device=args.device, shuffle=args.shuffle, k=args.k)
            for b in tqdm(range(n_batch), desc="Epoch %d" % (epoch+1)):
                optimizer.zero_grad()
                u, i, j = train.next_batch(b, model=self, k=args.k, eps=args.eps, opt=optimizer, choice=args.choice, grid=grid)
                # forward
                r_i = self.forward(u, i)
                reg_u, reg_i = self.reg_loss(args.l2)
                r_j = self.forward(u, j)
                reg_u, reg_j = self.reg_loss(args.l2)
                # log loss
                loss = - (r_i - r_j).sigmoid().log().sum() - args.adv * (r_i - r_j_).sigmoid().log().sum() + reg_u + reg_i + reg_j
                # gradients
                loss.backward()
                # update
                optimizer.step()
                loss_meter.update(loss.item())
                    
            # validation
            print("training loss:", loss_meter.avg)
            print("epoch negative quality:", neg_quality.avg)

            loss_meter.reset()
            neg_quality.reset()

    def evaluate(self, dataset, args):
        t = time.time()
        # eval mode
        self.eval()
        # loss
        criterion = args.loss
        # result
        result = [[], [], [], [], []]
        fast_res = []
        dataloader = dataset.build(batch_size=100, threshold=args.threshold, device=args.device)
        for u, i, r in dataloader:
            r_ = self.forward(u, i)
            loss = criterion(r_, r)
            fast_res.append(r_.detach().cpu().numpy())
            # result
            # for i, v in enumerate([u, i, r, r_]):
            #     if type(v) == type((1,2)): v = v[0]
            #     result[i].extend(list(v.cpu().detach().numpy()))
            # result[-1].append(loss.item())
        # fast_res = [metrics_fast(r, args.ks) for r in fast_res]
        pool = Pool(args.eval_workers)
        fast_res = pool.map(metrics_fast, fast_res)
        pool.close()
        pool.join()
        if args.metrics is None:
            request = ["loss", "mse", "mae", "rmse"] if args.threshold is None else ["loss", "ndcg", "precision", "recall", "auc"]
        else:
            request = args.metrics
        # performance = metrics(result, request)
        names = ["ndcg@%d" % k_ for k_ in args.ks] + ["hr@%d" % k_ for k_ in args.ks] + ["auc"] + ["pos"]
        variance = [names, np.var(fast_res, axis=0)]
        print("Var [%.2f]" % (time.time()-t) + " | ".join(["%s: %.4f" % (r, v) for r, v in zip(*variance)]))
        performance = [names, np.mean(fast_res, axis=0)]
        print("Acc [%.2f]" % (time.time()-t) + " | ".join(["%s: %.4f" % (r, v) for r, v in zip(*performance)]))
        
        # save the best parameter
        if "save_dir" in args.__dict__:
            # saving
            if "best" not in self.__dict__:
                flag = True
            elif request[0] in ["loss", "mse", "mae", "rmse"]:
                flag = bool(performance[1][0] < self.best)
            else:
                flag = bool(performance[1][0] > self.best)
            
            if flag:
                self.best = performance[1][0]
                # add time stamp
                time_local = time.localtime(int(time.time()))
                time_local = time.strftime("%Y-%m-%d-%H-%M-%S",time_local)
                # dir
                path = os.path.join(args.save_dir, time_local+"_epoch"+str(self.epoch)+"_"+"_".join(["%s%.4f" % (r, v) for r, v in zip(*performance)]))
                os.makedirs(path)
                # hyper parameters
                with open(os.path.join(path, "parameters.txt"), "w") as f:
                    for k, v in args.__dict__.items():
                        f.write("%s: %s\n" % (k, v))
                # model
                torch.save(self.state_dict(), os.path.join(path, "model.p"))
                print("==> save to %s ..." % path)
            
        return result


class VMF(MF):
    """ Follow the architecture of VBPR """
    def __init__(self, args):
        super(VMF, self).__init__(args)

        # mapping to orginal vbpr:
        # visual_user --> theta_user
        # visual_item_projector --> E
        # visual_bias --> beta_cnn
        # embedding_user, _item --> gamma_user, _item
        # bias_item --> beta_item

        # initialize the parameters
        self.visual_input_size = getattr(args, "img_input_size")
        self.visual_output_size = getattr(args, "img_output_size")

        # initialize item and user embedding
        # "+1" for some scnario that using padding index, or starting from 1
        self.visual_user = nn.Embedding(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.hidden_dim
        )
        self.visual_item_projector = \
            nn.Parameter(torch.randn(self.hidden_dim, self.visual_input_size).normal_(0, .01)) # (h, v)

        if args.biased:
            self.visual_bias = \
                 nn.Parameter(torch.randn(1, self.visual_input_size).normal_(0, .01)) # (1, v)

        self.visual_user.weight.data.normal_(0, .01)
        # self.visual_user.weight.data.uniform_()

    def init(self):

        # !!Embedding initialization below is very important!!
        # N(0,1) ruins our model for rating, use N(0,.1) instead
        # self.embed_user.weight.data.uniform_()
        # self.embed_item.weight.data.uniform_()

        self.embed_user.weight.data.normal_(0, .01)
        self.embed_item.weight.data.normal_(0, .01)

        try:
            self.bias_user.weight.data.fill_(0)
            self.bias_item.weight.data.fill_(0)
            self.bias = torch.tensor(self.args.global_mean) # can't learn

        except:
            pass

    def forward(self, user_indices, item_info):
        item_indices, item_visual = item_info
        # user embeddings
        self.p_u = self.embed_user(user_indices)
        # item embeddings
        self.q_i = self.embed_item(item_indices) 
        # # user visual embeddings
        self.v_u = self.visual_user(user_indices)
        # # item visual embeddings
        self.v_i = torch.matmul(item_visual, self.visual_item_projector.T) # (b, v) * (v, h)

        # print("v_i:", v_i[1])

        if self.args.biased:
            # user bias
            self.b_u = self.bias_user(user_indices).squeeze()
            # item bias
            self.b_i = self.bias_item(item_indices).squeeze()
            # visual bias
            self.b_v = torch.matmul(item_visual, self.visual_bias.T).squeeze() # (b, v) * (v, 1)
            # model
            return (self.p_u * self.q_i).sum(-1) \
                    + (self.v_u * self.v_i).sum(-1) \
                        + self.b_u + self.b_i + self.b_v + self.bias 
            # return (p_u * q_i).sum(-1) + b_u + b_i  + self.bias
        
        return (self.p_u * self.q_i).sum(-1) \
                + (self.v_u * self.v_i).sum(-1)


class GMF(MF):
    """ Matrix Factorization Model for ReAL quick experiment
        Follow the idea of y' = pq
    """
    def __init__(self, args):
        super(GMF, self).__init__(args)
        self.fc = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, user_indices, item_indices):
        # user embeddings
        self.p_u = self.embed_user(user_indices)
        # item embeddings
        self.q_i = self.embed_item(item_indices) 

        return self.fc(self.p_u * self.q_i).squeeze()

    def reg_loss(self, l2):
        reg_u = 0
        reg_i = 0
        # if type(l2) != dict:
        #     if l2 == 0: return 0, 0
        #     reg_w = sum((((self.mlp[i*2].weight)**2).sum() + (self.mlp[i*2].bias**2).sum()) for i in range(len(self.mlp)//2))
        #     if self.args.biased:
        #         return l2 * ((self.p_u**2).sum() + (self.b_u**2).sum() + reg_w), \
        #             l2 * ((self.p_u**2).sum() + (self.b_i**2).sum())
        #     else:
        #         return l2 * ((self.p_u**2).sum() + reg_w), l2 * (self.q_i**2).sum()
        for n, i in l2.items():
            if "p_u" in n: reg_u += i * (self.p_u**2).sum()
            if "q_i" in n: reg_i += i * (self.q_i**2).sum()
            if "b_u" in n: reg_u += i * (self.b_u**2).sum()
            if "b_i" in n: reg_i += i * (self.b_i**2).sum()
            if "fc" in n: reg_u += i * (self.fc.weight**2).sum()
        return reg_u, reg_i


class MLP(MF):
    def __init__(self, args):
        super(MLP, self).__init__(args)

        self.embed_user = torch.nn.Embedding(
            num_embeddings=self.num_users+1, 
            embedding_dim=args.layers[0]//2
        )
        self.embed_item = torch.nn.Embedding(
            num_embeddings=self.num_items+1, 
            embedding_dim=args.layers[0]//2
        )

        mlp = []
        layer1 = args.layers[0]
        for layer in args.layers[1:]:
            mlp.extend([nn.Linear(layer1, layer), nn.ReLU()])
            layer1 = layer
        self.mlp = torch.nn.Sequential(*mlp)
        self.fc = nn.Linear(layer, 1, bias=False)
        # init
        self.init()

    def forward(self, user_indices, item_indices, adv=None):
        # user embeddings
        self.p_u = self.embed_user(user_indices)
        # item embeddings
        self.q_i = self.embed_item(item_indices) 
        # adv training
        self.q_i = self.q_i + adv if adv is not None else self.q_i
        # mlp forward
        return self.fc(self.mlp(torch.cat((self.p_u, self.q_i), dim=1))).squeeze()
    
    def reg_loss(self, l2):
        reg_u = 0
        reg_i = 0
        if type(l2) != dict:
            if l2 == 0: return 0, 0
            reg_w = sum((((self.mlp[i*2].weight)**2).sum() + (self.mlp[i*2].bias**2).sum()) for i in range(len(self.mlp)//2))
            if self.args.biased:
                return l2 * ((self.p_u**2).sum() + (self.b_u**2).sum() + reg_w), \
                    l2 * ((self.p_u**2).sum() + (self.b_i**2).sum())
            else:
                return l2 * ((self.p_u**2).sum() + reg_w), l2 * (self.q_i**2).sum()
        for n, i in l2.items():
            if "p_u" in n: reg_u += i * (self.p_u**2).sum()
            if "q_i" in n: reg_i += i * (self.q_i**2).sum()
            if "b_u" in n: reg_u += i * (self.b_u**2).sum()
            if "b_i" in n: reg_i += i * (self.b_i**2).sum()
            for j in range(len(self.mlp)//2):
                if f"w_{j}" in n:
                    reg_u += i * (((self.mlp[j*2].weight)**2).sum() + (self.mlp[j*2].bias**2).sum())
            if "fc" in n: reg_u += i * (self.fc.weight**2).sum()
        return reg_u, reg_i

class NCF_NAIVE(MF):
    def __init__(self, args):
        super(NCF, self).__init__(args)
        # store args
        self.args = args
        self.epoch = 0
        # initialize the parameters
        self.num_users = getattr(args, "num_users")
        self.num_items = getattr(args, "num_items")
        self.hidden_dim = getattr(args, "hidden_dim")
        # gmf
        self.embed_user_gmf = torch.nn.Embedding(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.hidden_dim
        )
        self.embed_item_gmf = torch.nn.Embedding(
            num_embeddings=self.num_items+1, 
            embedding_dim=self.hidden_dim
        )
        # mlp
        self.embed_user_mlp = torch.nn.Embedding(
            num_embeddings=self.num_users+1, 
            embedding_dim=args.layers[0]//2
        )
        self.embed_item_mlp = torch.nn.Embedding(
            num_embeddings=self.num_items+1, 
            embedding_dim=args.layers[0]//2
        )
        # init
        self.embed_user_gmf.weight.data.normal_(0, .01)
        self.embed_item_gmf.weight.data.normal_(0, .01)
        self.embed_user_mlp.weight.data.normal_(0, .01)
        self.embed_item_mlp.weight.data.normal_(0, .01)
        # mlp layers
        mlp = []
        layer1 = args.layers[0]
        for layer in args.layers[1:]:
            mlp.extend([nn.Linear(layer1, layer), nn.ReLU()])
            layer1 = layer
        self.mlp = torch.nn.Sequential(*mlp)
        self.fc = nn.Linear(layer+self.hidden_dim, 1, bias=False)
        self.fc_gmf =  nn.Linear(self.hidden_dim, 1, bias=False)
        self.fc_mlp =  nn.Linear(layer, 1, bias=False)

    def forward(self, user_indices, item_indices):
        # gmf user embeddings
        self.p_u_gmf = self.embed_user_gmf(user_indices)
        # gmf item embeddings
        self.q_i_gmf = self.embed_item_gmf(item_indices) 
        # mlp user embeddings
        self.p_u_mlp = self.embed_user_mlp(user_indices)
        # mlp item embeddings
        self.q_i_mlp = self.embed_item_mlp(item_indices) 
        # mlp forward
        mlp = self.mlp(torch.cat((self.p_u_mlp, self.q_i_mlp), dim=1))
        # gmf = self.p_u_gmf * self.q_i_gmf
        # result
        return self.fc_mlp(mlp).squeeze()
        # return self.fc(torch.cat((mlp, gmf), dim=1)).squeeze()

    def reg_loss(self, l2):
        reg_u = 0
        reg_i = 0
        if type(l2) != dict:
            if l2 == 0: return 0, 0
            else:
                return l2 * ((self.p_u_gmf**2).sum() + (self.p_u_mlp**2).sum()), \
                    l2 * ((self.q_i_gmf**2).sum() + (self.q_i_mlp**2).sum())
        for n, i in l2.items():
            if "p_u_gmf" in n: reg_u += i * (self.p_u_gmf**2).sum()
            if "p_u_mlp" in n: reg_u += i * (self.p_u_mlp**2).sum()
            if "q_i_gmf" in n: reg_i += i * (self.q_i_gmf**2).sum()
            if "q_i_mlp" in n: reg_i += i * (self.q_i_mlp**2).sum()
        return reg_u, reg_i

class NCF(MF):
    """ 
    Concatate gmf and mlp embeddings at beginning, 
    equivalent to NCF_NAIVE but more convenient for ANS implimentation.
    """
    def __init__(self, args):
        super(NCF, self).__init__(args)
        # initialize the parameters
        self.gmf_dim = getattr(args, "hidden_dim") 
        self.mlp_dim = (args.layers[0]//2)
        self.hidden_dim = self.gmf_dim + self.mlp_dim
        # gmf
        self.embed_user = torch.nn.Embedding(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.hidden_dim
        )
        self.embed_item = torch.nn.Embedding(
            num_embeddings=self.num_items+1, 
            embedding_dim=self.hidden_dim
        )
        # init
        self.embed_user.weight.data.normal_(0, .01)
        self.embed_item.weight.data.normal_(0, .01)
        # mlp layers
        mlp = []
        layer1 = 2 * self.mlp_dim
        for layer in args.layers[1:]:
            mlp.extend([nn.Linear(layer1, layer), nn.ReLU()])
            layer1 = layer
        self.mlp = torch.nn.Sequential(*mlp)
        self.fc = nn.Linear(layer+self.gmf_dim, 1, bias=False)
        # init
        self.init()

    def forward(self, user_indices, item_indices):
        # embeddings 
        self.p_u = self.embed_user(user_indices)
        self.q_i = self.embed_item(item_indices) 
        # gmf user embeddings
        self.p_u_gmf = self.p_u.narrow(dim=1, start=0, length=self.gmf_dim)
        # gmf item embeddings
        self.q_i_gmf = self.q_i.narrow(dim=1, start=0, length=self.gmf_dim)
        # gmf forward
        gmf = self.p_u_gmf * self.q_i_gmf
        # mlp user embeddings
        self.p_u_mlp = self.p_u.narrow(dim=1, start=self.gmf_dim, length=self.mlp_dim)
        # mlp item embeddings
        self.q_i_mlp = self.q_i.narrow(dim=1, start=self.gmf_dim, length=self.mlp_dim)
        # mlp forward
        mlp = self.mlp(torch.cat((self.p_u_mlp, self.q_i_mlp), dim=1))
        # result
        return self.fc(torch.cat((gmf, mlp), dim=1)).squeeze()

    def reg_loss(self, l2):
        reg_u = 0
        reg_i = 0
        if type(l2) != dict:
            if l2 == 0: return 0, 0
            reg_w = sum(((self.mlp[i*2].weight)**2).sum() for i in range(len(self.mlp)//2)) + (self.fc.weight**2).sum()
            if self.args.biased:
                return l2 * ((self.p_u**2).sum() + (self.b_u**2).sum() + reg_w), \
                    l2 * ((self.p_u**2).sum() + (self.b_i**2).sum())
            else:
                return l2 * ((self.p_u**2).sum() + reg_w), l2 * (self.q_i**2).sum()
        for n, i in l2.items():
            if "p_u" in n: reg_u += i * (self.p_u_gmf**2).sum()
            if "q_i" in n: reg_i += i * (self.q_i_gmf**2).sum()
            if "b_u" in n: reg_u += i * (self.b_u**2).sum()
            if "b_i" in n: reg_i += i * (self.b_i**2).sum()
            for j in range(len(self.mlp)//2):
                if f"w_{j}" in n:
                    reg_u += i * (((self.mlp[j*2].weight)**2).sum() + (self.mlp[j*2].bias**2).sum())
            if "fc" in n: reg_u += i * (self.fc.weight**2).sum()
        return reg_u, reg_i

class ConvNCF(MF):
    def __init__(self, args):
        super(ConvNCF, self).__init__(args)
        self.kernel_size = 2
        self.num_conv = int(np.log2(self.hidden_dim)) - 1
        self.conv = nn.Sequential( 
            *([nn.Conv2d(1, 32, self.kernel_size, stride=2), nn.ReLU()] \
                + ([nn.Conv2d(32, 32, self.kernel_size, stride=2), nn.ReLU()]) * self.num_conv)
        )
        self.fc = nn.Linear(32, 1)
        self.conv_init()

    def conv_init(self):
        for i in range(len(self.conv)//2):
            self.conv[i*2].weight.data.normal_(0, .1)
            self.conv[i*2].bias.data.fill_(0.1)
        self.fc.weight.data.normal_(0, .1)
        self.fc.bias.data.fill_(0.1)

    def forward(self, user_indices, item_indices):
        # user embeddings
        self.p_u = self.embed_user(user_indices) # (batch, k)
        # item embeddings
        self.q_i = self.embed_item(item_indices) # (batch, k)
        # interaction maps
        maps = torch.bmm(self.p_u.unsqueeze(2), self.q_i.unsqueeze(1)) # (batch, k, k) where k = 64
        maps = maps.unsqueeze(1)
        # conv & ReLU
        x = self.conv(maps)
        x = self.fc(x.squeeze())

        return x.squeeze()

    def reg_loss(self, l2):
        reg_u = 0
        reg_i = 0
        if type(l2) != dict:
            if l2 == 0: return 0, 0
            reg_w = sum(((self.conv[i*2].weight)**2).sum() for i in range(len(self.conv)//2)) + (self.fc.weight**2).sum()
            if self.args.biased:
                return l2 * ((self.p_u**2).sum() + (self.b_u**2).sum() + reg_w), \
                    l2 * ((self.p_u**2).sum() + (self.b_i**2).sum())
            else:
                return l2 * ((self.p_u**2).sum() + reg_w), l2 * (self.q_i**2).sum()
        for n, i in l2.items():
            if "p_u" in n: reg_u += i * (self.p_u**2).sum()
            if "q_i" in n: reg_i += i * (self.q_i**2).sum()
            if "b_u" in n: reg_u += i * (self.b_u**2).sum()
            if "b_i" in n: reg_i += i * (self.b_i**2).sum()
            if "c" in n: reg_u += i * sum(((self.conv[i*2].weight)**2).sum() for i in range(len(self.conv)//2))
            if "w" in n: reg_u += i * (self.fc.weight**2).sum()
        return reg_u, reg_i
