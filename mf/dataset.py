import random
import numpy as np
import pandas as pd
import warnings
import torch
import faiss
import collections

from tqdm import tqdm
from time import time
from pathos.multiprocessing import ProcessingPool as Pool

from utils import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_LOSS_COL,
    random_choice
)

class Dataset(object):
    """Dataset class for ReAL Training set"""

    def __init__(
        self,
        train,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        seed=None,
    ):
        """Constructor 
        
        Args:
            train (pd.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pd.DataFrame): Test data with at least columns (col_user, col_item, col_rating). test can be None, 
                if so, we only process the training data.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name. 
            col_timestamp (str): Timestamp column name.
            binary (bool): If true, set rating > 0 to rating = 1. 
            seed (int): Seed.
        
        """

        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        # dataset
        self.train = train
        self.train_np = train.to_numpy()[:, :3]
        # initialize negative sampling for training
        self._init_train_data()
        # set random seed
        random.seed(seed)

    def _init_train_data(self):
        self.item_pool_list = list(self.train[self.col_item].unique())
        self.item_pool = set(self.item_pool_list)
        interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(list)
            .reset_index()
            .rename(columns={self.col_item: str(self.col_item) + "_interacted"})
        )
        self.interact_status_list = dict(zip(list(interact_status[self.col_user]), \
                list(interact_status[str(self.col_item) + "_interacted"])))
        self.interact_status = {k:set(v) for k, v in self.interact_status_list.items()}

        self.users, self.items, self.ratings = \
            self.train[self.col_user].values, self.train[self.col_item].values, self.train[self.col_rating].values

        # self.freq_list = [list(self.items).count(i) for i in self.item_pool_list]
        freq_dict = collections.Counter(self.items)
        self.freq_list = [e[0] for e in freq_dict.most_common()] # the items ranked in terms of frequency

    def _negative_sampling(self, n_neg):
        self.users, self.items, self.ratings = [], [], []

        # generate training data
        for row in self.train.itertuples():
            user = int(row[self.col_user+1])
            self.users.append(user)
            self.items.append(int(row[self.col_item+1]))
            self.ratings.append(float(row[self.col_rating+1]))
            self.users.extend([int(row[self.col_user+1])]*n_neg)
            items_neg = []
            while len(items_neg) < n_neg:
                item_neg = random.choice(self.item_pool_list)
                if item_neg not in self.interact_status[user]:
                    items_neg.append(item_neg)
            self.items.extend(items_neg)
            self.ratings.extend([0.]*n_neg)

        self.users = np.array(self.users)
        self.items = np.array(self.items)
        self.ratings = np.array(self.ratings)

    def _shuffle(self, lists):
        indices = list(range(len(lists[0])))
        random.shuffle(indices)
        return [l[indices] for l in lists]


    def build(self, n_neg=0, batch_size=32, shuffle=True, threshold=None, device="cuda"):
        import torch
        if n_neg:
            self._negative_sampling(n_neg)

        res = (self.users, self.items, self.ratings)
        res = self._shuffle(res) if shuffle else res

        users, items, ratings = torch.LongTensor(res[0]),  torch.LongTensor(res[1]),  torch.Tensor(res[2])

        if threshold is not None:
            ratings = (ratings >= threshold).float()

        if device == "cuda":
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u, i, r = users[k*batch_size: (k+1)*batch_size], items[k*batch_size: (k+1)*batch_size], ratings[k*batch_size: (k+1)*batch_size]
            yield (u, i, r)

class FastDataset(object):
    def __init__(
        self,
        test,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        seed=None,
    ):
        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        # dataset
        self.test = test
        # initialize negative sampling for training
        self._init_test_data()
        # set random seed
        random.seed(seed)

    def _init_test_data(self):
        self.test = self.test.sort_values(by=[self.col_user, self.col_rating], ascending=False).reset_index(drop=True)
        self.batch_size = len(self.test) // self.test[self.col_user].nunique()
        print("Batch size of FastDataset: ", self.batch_size)
        self.users = self.test[self.col_user].tolist()
        self.items = self.test[self.col_item].tolist()
        self.ratings = self.test[self.col_rating].tolist()

    def build(self, batch_size=100, threshold=1, device=None):
        batch_size = self.batch_size
        users, items, ratings = torch.LongTensor(self.users),  torch.LongTensor(self.items),  torch.Tensor(self.ratings)

        if threshold is not None:
            ratings = (ratings >= threshold).float()

        if device == "cuda":
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u, i, r = users[k*batch_size: (k+1)*batch_size], items[k*batch_size: (k+1)*batch_size], ratings[k*batch_size: (k+1)*batch_size]
            assert u[0] == u[-1]
            yield (u, i, r)

class POPDataset(Dataset):
    def __init__(
        self,
        train,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        seed=None
    ):
        super().__init__(train, col_user=col_user, col_item=col_item, col_rating=col_rating, col_timestamp=col_timestamp, seed=seed)
        self.item_count = sorted(collections.Counter(self.items).items(), key=lambda c: c[1])
        self.user_count = collections.Counter(self.users)
        self.freq_prob = np.zeros(len(self.items))
        for i, c in enumerate(self.item_count):
            self.freq_prob[c[0]] = np.exp(-i) / 10

    def _negative_sampling(self, n_neg=1):
        self.users, self.items_positive, self.items_negative = [], [], []

        # generate training data by users
        for user in tqdm(self.user_count, desc="Sampling..."):
            n_user = self.user_count[user]
            self.users.extend([user]*n_user)
            self.items_positive.extend(self.interact_status_list[user])
            freq_prob = self._get_neg_prob(user)
            # accelerate with vectorizing
            self.items_negative.extend(np.random.choice(self.items, n_user, p=freq_prob, replace=True))

        self.users = np.array(self.users)
        self.items_positive = np.array(self.items_positive)
        self.items_negative = np.array(self.items_negative)

        print(len(self.users), len(self.items_positive), len(self.items_negative))

    def _get_neg_prob(self, user):
        prob = np.array(self.freq_prob, copy=True)
        prob[list(self.interact_status_list[user])] = 0
        return prob / np.sum(prob)
                                
    def build(self, n_neg=1, batch_size=32, shuffle=True, device="cuda"):
        import torch
        self._negative_sampling(n_neg)
        res = (self.users, self.items_positive, self.items_negative)
        res = self._shuffle(res) if shuffle else res

        users, items_positive, items_negative = torch.LongTensor(res[0]),  torch.LongTensor(res[1]),  torch.LongTensor(res[2])
        if device == "cuda":
            users, items_positive, items_negative = users.cuda(), items_positive.cuda(), items_negative.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u, i, j = users[k*batch_size: (k+1)*batch_size], items_positive[k*batch_size: (k+1)*batch_size], items_negative[k*batch_size: (k+1)*batch_size]
            yield (u, i, j)

class POP2Dataset(Dataset):
    def build(self, batch_size=32, shuffle=True, device="cuda", k=10):
        self.batch_size = batch_size
        self.device = device
        self.ones = torch.ones(batch_size)
        if self.device == "cuda":
            self.ones = self.ones.cuda()
        if shuffle:
            self.train = self.train.sample(len(self.train))
            self.train_np = self.train.to_numpy()[:, :3]
        item_count = sorted(collections.Counter(self.items).items(), key=lambda c: c[1])
        self.freq_prob = np.zeros(len(self.items))
        for i, c in enumerate(item_count):
            self.freq_prob[c[0]] = np.exp(-i) / 1
        return len(self.train) // batch_size

    def _get_neg_prob(self, user):
        prob = np.array(self.freq_prob, copy=True)
        prob[list(self.interact_status[user])] = 0
        return prob / np.sum(prob)

    def vectorized(self, prob_matrix, items):
        s = prob_matrix.cumsum(axis=0)
        r = np.random.rand(prob_matrix.shape[1])
        k = (s < r).sum(axis=0)
        return items[k]

    def next_batch(self, batch, model, k=10, eps=None, opt=None, choice="hard", bp=None, l2=0):

        batch = self.train.iloc[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, prob_matrix, items_negative = [], [], [], []

        # generate training data
        for row in batch.itertuples():
            user = int(row[self.col_user+1])
            users.append(user)
            items_positive.append(int(row[self.col_item+1]))
            items_negative.append(np.random.choice(self.items, p=self._get_neg_prob(user)))
            # prob_matrix.append(self._get_neg_prob(user))
        
        # vectorized sampling
        # items_negative = self.original(np.array(prob_matrix), self.items)

        users, items_positive, items_negative = \
            torch.LongTensor(users),  torch.LongTensor(items_positive),  torch.LongTensor(items_negative)

        if self.device == "cuda":
            users, items_positive, items_negative = users.cuda(), items_positive.cuda(), items_negative.cuda()

        return (users, items_positive, items_negative)

class BPRDataset(Dataset):
    def _negative_sampling(self, n_neg=1):
        self.users, self.items_positive, self.items_negative = [], [], []
        #print(self.freq_list)

        # generate training data
        for row in self.train.itertuples():
            user = int(row[self.col_user+1])
            self.users.append(user)
            self.items_positive.append(int(row[self.col_item+1]))
            for _ in range(n_neg):
                candidate = random.choice(self.item_pool_list)
                while candidate in self.interact_status[user]:
                    candidate = random.choice(self.item_pool_list)
                self.items_negative.append(candidate)

        self.users = np.array(self.users)
        self.items_positive = np.array(self.items_positive)
        self.items_negative = np.array(self.items_negative)

    def build(self, n_neg=1, batch_size=32, shuffle=True, device="cuda"):
        import torch

        self._negative_sampling(n_neg)

        res = (self.users, self.items_positive, self.items_negative)
        res = self._shuffle(res) if shuffle else res

        users, items_positive, items_negative = torch.LongTensor(res[0]),  torch.LongTensor(res[1]),  torch.LongTensor(res[2])
        if device == "cuda":
            users, items_positive, items_negative = users.cuda(), items_positive.cuda(), items_negative.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u, i, j = users[k*batch_size: (k+1)*batch_size], items_positive[k*batch_size: (k+1)*batch_size], items_negative[k*batch_size: (k+1)*batch_size]
            yield (u, i, j)

class POPDataset(Dataset):
    def _negative_sampling(self, n_neg=1):
        self.users, self.items_positive, self.items_negative = [], [], []
        #print(self.freq_list)
        self.freq_prob = [np.exp(-self.freq_list.index(a)) for a in self.freq_list]
        sum_freq = sum(self.freq_prob)
        self.freq_prob = [a/sum_freq for a in self.freq_prob]
        #print(self.freq_prob)
        # generate training data
        for row in self.train.itertuples():
            user = int(row[self.col_user+1])
            self.users.append(user)
            self.items_positive.append(int(row[self.col_item+1]))
            for _ in range(n_neg):
                candidate = random.choices(self.item_pool_list, weights=self.freq_prob)[0]
                while candidate in self.interact_status[user]:
                    candidate = random.choices(self.item_pool_list, weights=self.freq_prob)[0]
                self.items_negative.append(candidate)

        self.users = np.array(self.users)
        self.items_positive = np.array(self.items_positive)
        self.items_negative = np.array(self.items_negative)

    def build(self, n_neg=1, batch_size=32, shuffle=True, device="cuda"):
        import torch

        self._negative_sampling(n_neg)

        res = (self.users, self.items_positive, self.items_negative)
        res = self._shuffle(res) if shuffle else res

        users, items_positive, items_negative = torch.LongTensor(res[0]),  torch.LongTensor(res[1]),  torch.LongTensor(res[2])
        if device == "cuda":
            users, items_positive, items_negative = users.cuda(), items_positive.cuda(), items_negative.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u, i, j = users[k*batch_size: (k+1)*batch_size], items_positive[k*batch_size: (k+1)*batch_size], items_negative[k*batch_size: (k+1)*batch_size]
            yield (u, i, j)


class DNSDataset(Dataset):
    def build(self, batch_size=32, shuffle=True, device="cuda", k=10):
        self.batch_size = batch_size
        self.device = device
        self.ones = torch.ones(batch_size)
        if self.device == "cuda":
            self.ones = self.ones.cuda()
        if shuffle:
            self.train = self.train.sample(len(self.train))
            self.train_np = self.train.to_numpy()[:, :3]
        sort = np.arange(k)
        self.p = sort / sort.sum()
        self.item_size = len(self.item_pool_list)
        return len(self.train) // batch_size

    def next_batch(self, batch, model, k=10, eps=None, opt=None, choice="hard", bp=None, l2=0):

        batch = self.train.iloc[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, items_negative = [], [], []

        # generate training data
        for row in batch.itertuples():
            user = int(row[self.col_user+1])
            users.append(user)
            items_positive.append(int(row[self.col_item+1]))
            candidates = []
            while len(candidates) < k:
                candidate = random.choice(self.item_pool_list)
                if candidate not in self.interact_status[user]:
                    candidates.append(candidate)

            user = torch.LongTensor([user]*k)
            item = torch.LongTensor(candidates)
            if self.device == "cuda":
                user, item = user.cuda(), item.cuda()
            r_ = model.forward(user, item)

            if choice == "hard":
                items_negative.append(candidates[torch.argmax(r_).item()])
            elif choice == "soft":
                candidtes = np.array(candidates)[r_.cpu().argsort()]
                items_negative.extend(random.choices(candidates, weights=self.p))
            else:
                sim = r_.detach().cpu().numpy()
                sim -= min(sim) - 1e-7
                items_negative.extend(random.choices(candidates, sim/sim.sum()))
        
        users, items_positive, items_negative = \
            torch.LongTensor(users),  torch.LongTensor(items_positive),  torch.LongTensor(items_negative)

        if self.device == "cuda":
            users, items_positive, items_negative = users.cuda(), items_positive.cuda(), items_negative.cuda()

        return (users, items_positive, items_negative)

class ANSDataset(DNSDataset):
    def knn(self, target, user, item, candidates, k=None, eps=float("inf"), t=None, choice="hard"):
        j = candidates[item]
        random_index = [item]
        unhit = 0
        while len(random_index) < k:
            index = random.choice(self.item_pool_list)
            # if (index not in self.interact_status[user]) and (max(torch.abs(candidates[index] - j)) <= eps):
            #     random_index.append(index)
            # else:
            #     unhit += 1
            #     if unhit > 10*k: 
            #         print("len(random_index): ", len(random_index))
            #         break
            if (index not in self.interact_status[user]):
                random_index.append(index)
        if len(random_index) == 1: 
            return random_index[0]
        candidates = candidates[random_index]
        # actual embedding search
        sim = torch.matmul(candidates, target)
        # sampling
        if choice == "hard":
            return [random_index[torch.argmax(sim).item()]]
        elif choice == "soft":
            random_index = np.array(random_index)[sim.cpu().argsort()]
            return random.choices(random_index, weights=self.p)
        else:
            sim = sim.detach().cpu().numpy()
            sim -= min(sim) - 1e-7
            return random.choices(random_index, weights=sim/sim.sum())

    def adv(self, loss, model, candiate):
        loss.backward(self.ones, retain_graph=True)
        return model.embed_item.weight.grad[candiate]

    def next_batch(self, batch, model, k=None, eps=None, opt=None, choice="hard", bp=False, l2=0):
        # batch size = 1
        batch = self.train.iloc[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, items_candidate, items_negative = [], [], [], []

        # generate training data
        for row in batch.itertuples():
            user = int(row[self.col_user+1])
            users.append(user)
            items_positive.append(int(row[self.col_item+1]))
            candidate = random.choice(self.item_pool_list)
            while candidate in self.interact_status[user]:
                candidate = random.choice(self.item_pool_list)
            items_candidate.append(candidate)
            items_negative.append(candidate)

        users_ = torch.LongTensor(users)
        items_j = torch.LongTensor(items_candidate)
        if bp:
            items_i = torch.LongTensor(items_positive)
            if self.device == "cuda":
                users_, item_i, items_j = users_.cuda(), item_i.cuda(), items_j.cuda()
            r_i = model.forward(users_, items_i)
            reg_u, reg_i = model.reg_loss(l2)
            r_j = model.forward(users_, items_j)
            reg_u, reg_j = model.reg_loss(l2)
            adv = self.adv(-(r_i-r_j).sigmoid().log(), model, items_candidate)
        else:
            if self.device == "cuda":
                users_, items_j = users_.cuda(), items_j.cuda()
            r_j = model.forward(users_, items_j)
            adv = self.adv(-(-r_j).sigmoid().log(), model, items_candidate)

        items_better_negative = []

        for c, (u, i) in enumerate(zip(users, items_candidate)):
            better_neg = \
                self.knn(adv[c], u, i, model.embed_item.weight, k=k, eps=eps, choice=choice)
            items_better_negative.extend(better_neg)
        if bp:
            if l2:
                reg = reg_u + reg_i + reg_j
                reg.backward()
            opt.step()
        opt.zero_grad()
        
        users, items_positive, items_better_negative = \
            torch.LongTensor(users), torch.LongTensor(items_positive), torch.LongTensor(items_better_negative)

        if self.device == "cuda":
            users, items_positive, items_better_negative = users.cuda(), items_positive.cuda(), items_better_negative.cuda()

        return (users, items_positive, items_better_negative)

class GANSDataset(DNSDataset):
    def knn(self, target, random_index, candidates, choice="hard"):
        if max(random_index) == min(random_index):
            return [random_index[0]]
        # multi candidates
        candidates = candidates[random_index]
        # actual embedding search
        sim = torch.matmul(candidates, target)
        # sampling
        if choice == "hard":
            return [random_index[torch.argmax(sim).item()]]
        elif choice == "soft":
            random_index = np.array(random_index)[sim.cpu().argsort()]
            return random.choices(random_index, weights=self.p)
        else:
            sim = sim.detach().cpu().numpy()
            sim -= min(sim) - 1e-7
            return random.choices(random_index, weights=sim/sim.sum())

    def adv(self, loss, model, candiate):
        loss.backward(self.ones)
        return model.embed_item.weight.grad[candiate]

    def next_batch(self, batch, model, k=None, eps=None, opt=None, choice="hard", grid=None, bp=None, l2=0):
        # batch size = 1
        batch = self.train_np[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, items_candidate, random_indices, items_better_negative = [], [], [], [], []

        # generate training data
        for row in batch:
            user = int(row[self.col_user])
            users.append(user)
            items_positive.append(int(row[self.col_item]))
            candidate = random.choice(self.item_pool_list)
            while candidate in self.interact_status[user]:
                candidate = random.choice(self.item_pool_list)
            items_candidate.append(candidate)

        # grid search candidates
        #######################################
        # TODO: search from positive items
        # random_index_batch = grid.search_by_id(items_positive)
        random_index_batch = grid.search_id_batch(items_candidate)
        length = [len(r) for r in random_index_batch]
        for i, (u, r) in enumerate(zip(users, random_index_batch)):
            while length[i] < 100:
                candidate = random.choice(self.item_pool_list)
                while candidate in self.interact_status[u]:
                    candidate = random.choice(self.item_pool_list)
                r = grid.search_id(candidate)
                length[i] = len(r)
            random_index = []
            while len(random_index) < k:
                temp = random.choice(r)
                if temp not in self.interact_status[u]:
                    random_index.append(temp)

            #######################################
            # TODO: search from positive items
            # r = set(r) - self.interact_status[u]
            # if len(r) == 0:
            #     j = items_candidate[i]
            #     # print(grid.map, j, model.embed_item.weight[j], grid.indexing(model.embed_item.weight[j]))
            #     random_index = [j]
            # else:
            #     random_index = list(r) if len(r) < k else random.sample(r, k)
            random_indices.append(random_index)

        # # adversarial searching
        users_ = torch.LongTensor(users)

        items_ = torch.LongTensor(items_candidate)
        #######################################
        # TODO: search from positive items
        # items_ = torch.LongTensor(items_positive)
        
        if self.device == "cuda":
            users_, items_ = users_.cuda(), items_.cuda()
        r_ = model.forward(users_, items_)
        #######################################
        # TODO: search from positive items
        # adv = self.adv(-(1-r_).sigmoid().log(), model, items_positive)
        adv = self.adv(-(1-r_).sigmoid().log(), model, items_candidate)
        opt.zero_grad()

        # adversarial sampling
        for c, r in enumerate(random_indices):
            better_neg = self.knn(adv[c], r, model.embed_item.weight, choice=choice)
            items_better_negative.extend(better_neg)
        
        # dataset generation
        users, items_positive, items_better_negative = \
            torch.LongTensor(users), torch.LongTensor(items_positive), torch.LongTensor(items_better_negative)

        if self.device == "cuda":
            users, items_positive, items_better_negative = users.cuda(), items_positive.cuda(), items_better_negative.cuda()

        return (users, items_positive, torch.LongTensor(items_candidate).cuda(), items_better_negative, length)

class FANSDataset(DNSDataset):
    def knn(self, target, random_index, candidates, choice="hard"):
        if max(random_index) == min(random_index):
            return [random_index[0]]
        # multi candidates
        candidates = candidates[random_index]
        # actual embedding search
        sim = torch.matmul(candidates, target)
        # sampling
        if choice == "hard":
            return [random_index[torch.argmax(sim).item()]]
        elif choice == "soft":
            random_index = np.array(random_index)[sim.cpu().argsort()]
            return random.choices(random_index, weights=self.p)
        else:
            sim = sim.detach().cpu().numpy()
            sim -= min(sim) - 1e-7
            return random.choices(random_index, weights=sim/sim.sum())

    def adv(self, loss, model, candiate):
        loss.backward(self.ones)
        return model.embed_item.weight.grad[candiate]

    def next_batch(self, batch, model, k=None, eps=None, opt=None, choice="hard", grid=None, bp=None, l2=0):
        # batch size = 1
        batch = self.train_np[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, items_candidate, random_indices, items_dns_negative, items_better_negative = [], [], [], [], [], []

        # generate training data
        for row in batch:
            user = int(row[self.col_user])
            users.append(user)
            items_positive.append(int(row[self.col_item]))
            candidate = random.choice(self.item_pool_list)
            while candidate in self.interact_status[user]:
                candidate = random.choice(self.item_pool_list)
            items_candidate.append(candidate)

        # adversarial searching
        users_ = torch.LongTensor(users)
        #######################################
        # TODO: search from positive items
        # items_ = torch.LongTensor(items_positive)
        items_ = torch.LongTensor(items_candidate)
        
        if self.device == "cuda":
            users_, items_ = users_.cuda(), items_.cuda()
        r_ = model.forward(users_, items_)
        #######################################
        # TODO: search from positive items
        # adv = self.adv(-(1-r_).sigmoid().log(), model, items_positive)
        adv = self.adv(-(1-r_).sigmoid().log(), model, items_candidate)
        adv = eps * adv / adv.norm(dim=1, keepdim=True)
        opt.zero_grad()

        # grid search candidates
        #######################################
        # TODO: search from positive items
        # _, random_index_batch = grid.search(model.embed_item.weight[items_positive].detach().cpu().numpy(), 2*k)

        #######################################
        # TODO: why two doesn't work
        _, knn_index_batch = grid.search((model.embed_item.weight[items_candidate] + adv).detach().cpu().numpy(), 10*k)

        length = [len(r) for r in knn_index_batch]
        for i, (u, r) in enumerate(zip(users, knn_index_batch)):
            random_index = []
            for temp in r:
                if len(random_index) >= k or temp == -1:
                    break
                if temp not in self.interact_status[u]:
                    random_index.append(temp)
            #######################################
            # TODO: search from positive items
            # r = set(r) - self.interact_status[u]
            # if len(r) == 0:
            #     j = items_candidate[i]
            #     # print(grid.map, j, model.embed_item.weight[j], grid.indexing(model.embed_item.weight[j]))
            #     random_index = [j]
            # else:
            #     random_index = list(r) if len(r) < k else random.sample(r, k)
            if len(random_index) == 0:
                print("0:")
                random_index = [items_candidate[i]]
            length[i] = len(random_index)
            random_indices.append(random_index)

        # adversarial sampling
        for c, r in enumerate(random_indices):
            better_neg = self.knn(adv[c], r, model.embed_item.weight, choice=choice)
            items_better_negative.extend(better_neg)

        random_index_batch = []
        for i, row in enumerate(batch):
            user = int(row[self.col_user])
            candidates = []
            while len(candidates) < (k-1):
                candidate = random.choice(self.item_pool_list)
                if candidate not in self.interact_status[user]:
                    candidates.append(candidate)
            random_index_batch.append([items_candidate[i]] + candidates)

        for c, r in enumerate(random_index_batch):
            better_neg = self.knn(adv[c], r, model.embed_item.weight, choice=choice)
            items_dns_negative.extend(better_neg)
        
        # dataset generation
        #######################################
        # TODO: search from positive items
        # users, items_positive = users_, items_
        # items_negative, items_better_negative = torch.LongTensor(items_candidate), torch.LongTensor(items_better_negative)
        users, items_negative = users_, items_
        items_positive, items_dns_negative, items_better_negative = torch.LongTensor(items_positive), torch.LongTensor(items_dns_negative), torch.LongTensor(items_better_negative)

        if self.device == "cuda":
            users, items_positive, items_negative, items_dns_negative, items_better_negative = users.cuda(), items_positive.cuda(), items_negative.cuda(), items_dns_negative.cuda(), items_better_negative.cuda()

        return (users, items_positive, items_negative, items_dns_negative, items_better_negative, length)

class TANSDataset(DNSDataset):
    def knn(self, target, random_index, candidates, choice="hard"):
        if max(random_index) == min(random_index):
            return [random_index[0]]
        # multi candidates
        candidates = candidates[random_index]
        # actual embedding search
        sim = torch.matmul(candidates, target)
        # sampling
        if choice == "hard":
            return [random_index[torch.argmax(sim).item()]]
        elif choice == "soft":
            random_index = np.array(random_index)[sim.cpu().argsort()]
            return random.choices(random_index, weights=self.p)
        else:
            sim = sim.detach().cpu().numpy()
            sim -= min(sim) - 1e-7
            return random.choices(random_index, weights=sim/sim.sum())

    def adv(self, loss, model, candiate):
        loss.backward(self.ones)
        return model.embed_item.weight.grad[candiate]

    def search(self, weights, items, users, k, eps):
        random_index_batch = []
        for u, i in zip(users, items):
            candidates = (torch.max(torch.abs(weights - weights[i]), dim=1)[0] <= eps).nonzero().squeeze().tolist()
            random_index = []
            while len(random_index) < k:
                index = random.choice(candidates)
                if index not in self.interact_status[u]:
                    random_index.append(index)
            random_index_batch.append(random_index)
        return random_index_batch

    def next_batch(self, batch, model, k=None, eps=None, opt=None, choice="hard", grid=None, bp=None, l2=0):
        # batch size = 1
        batch = self.train_np[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, items_candidate, random_indices, items_better_negative = [], [], [], [], []

        # generate training data
        for row in batch:
            user = int(row[self.col_user])
            users.append(user)
            items_positive.append(int(row[self.col_item]))
            candidate = random.choice(self.item_pool_list)
            while candidate in self.interact_status[user]:
                candidate = random.choice(self.item_pool_list)
            items_candidate.append(candidate)

        # grid search candidates
        #######################################
        # TODO: search from positive items
        # _, random_index_batch = grid.search(model.embed_item.weight[items_positive].detach().cpu().numpy(), 2*k)

        #######################################
        # TODO: why two doesn't work
        random_indices = self.search(model.embed_item.weight.detach(), items_candidate, users, k, eps)
        # random_index_batch = []
        # for i, row in enumerate(batch):
        #     user = int(row[self.col_user])
        #     candidates = []
        #     while len(candidates) < (k-1):
        #         candidate = random.choice(self.item_pool_list)
        #         if candidate not in self.interact_status[user]:
        #             candidates.append(candidate)
        #     random_index_batch.append([items_candidate[i]] + candidates)

        length = [len(r) for r in random_indices]
        # # adversarial searching
        users_ = torch.LongTensor(users)
        #######################################
        # TODO: search from positive items
        # items_ = torch.LongTensor(items_positive)
        items_ = torch.LongTensor(items_candidate)
        
        if self.device == "cuda":
            users_, items_ = users_.cuda(), items_.cuda()
        r_ = model.forward(users_, items_)
        #######################################
        # TODO: search from positive items
        # adv = self.adv(-(1-r_).sigmoid().log(), model, items_positive)
        adv = self.adv(-(1-r_).sigmoid().log(), model, items_candidate)
        opt.zero_grad()

        # adversarial sampling
        for c, r in enumerate(random_indices):
            better_neg = self.knn(adv[c], r, model.embed_item.weight, choice=choice)
            items_better_negative.extend(better_neg)
        
        # dataset generation
        #######################################
        # TODO: search from positive items
        # users, items_positive = users_, items_
        # items_negative, items_better_negative = torch.LongTensor(items_candidate), torch.LongTensor(items_better_negative)
        users, items_negative = users_, items_
        items_positive, items_better_negative = torch.LongTensor(items_positive), torch.LongTensor(items_better_negative)

        if self.device == "cuda":
            users, items_positive, items_negative, items_better_negative = users.cuda(), items_positive.cuda(), items_negative.cuda(), items_better_negative.cuda()

        return (users, items_positive, items_negative, items_better_negative, length)

class PDNSDataset(Dataset):
    def build(self, n_neg=1, batch_size=32, shuffle=True, device="cuda"):
        self.n_neg = n_neg
        self.batch_size = batch_size // (1+n_neg)
        self.device = device
        self.ones = torch.ones(self.batch_size)
        if self.device == "cuda":
            self.ones = self.ones.cuda()
        if shuffle:
            self.train = self.train.sample(len(self.train))
        return len(self.train) // self.batch_size

    def next_batch(self, batch, model, k=10, loss=None, eps=None, opt=None, choice="hard"):
        n_neg = self.n_neg

        batch = self.train_np[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items, ratings = [], [], []

        # generate training data
        for row in batch:
            user = int(row[self.col_user])
            users.append(user)
            items.append(int(row[self.col_item]))
            ratings.append(float(row[self.col_rating]>=1))
            candidates = []
            while len(candidates) < k:
                candidate = random.choice(self.item_pool_list)
                if candidate not in self.interact_status[user]:
                    candidates.append(candidate)
            # candidates = random.sample(self.item_pool-self.interact_status[user], k)

            user_ = torch.LongTensor([user]*k)
            item_ = torch.LongTensor(candidates)
            if self.device == "cuda":
                user_, item_ = user_.cuda(), item_.cuda()
            r_ = model.forward(user_, item_)

            if choice == "soft":
                sort = np.zeros(k)
                for i, j in enumerate(r_.cpu().argsort()):
                    sort[j.item()] = k - i - 1
                p = 1 - (sort/(k-1))
                p /= p.sum()
                items.extend(random.choices(candidates, p, k=n_neg))
                users.extend([user] * n_neg)
                ratings.extend([0] * n_neg)
            else:
                sim = r_.detach().cpu().numpy()
                sim -= min(sim) - 1e-7
                items.extend(random.choices(candidates, sim/sim.sum(), k=n_neg))
                users.extend([user] * n_neg)
                ratings.extend([0] * n_neg)

        assert len(items) == len(users)
        
        users, items, ratings = \
            torch.LongTensor(np.array(users)),  torch.LongTensor(np.array(items)), torch.Tensor(np.array(ratings)) 

        if self.device == "cuda":
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        return (users, items, ratings)

class PANSDataset(Dataset):
    def build(self, n_neg=1, batch_size=32, shuffle=True, device="cuda"):
        self.n_neg = n_neg
        self.batch_size = batch_size // (1+n_neg)
        self.device = device
        self.ones = torch.ones(self.batch_size)
        if self.device == "cuda":
            self.ones = self.ones.cuda()
        if shuffle:
            self.train = self.train.sample(len(self.train))
        return len(self.train) // self.batch_size
    
    def knn(self, target, random_index, candidates, n_neg, k=None, eps=float("inf"), t=None, choice="hard"):
        if len(random_index) == 1: 
            return [random_index[0]]
        # actual embedding search
        sim = torch.matmul(candidates, target)
        # sampling
        if choice == "hard":
            return random_index[torch.argmax(sim).item()]
        elif choice == "soft":
            sort = np.zeros(k)
            for i, j in enumerate(sim.cpu().argsort()):
                sort[j.item()] = k - i - 1
            p = 1 - (sort/(k-1))
            p /= p.sum()
            return random.choices(random_index, p, k=n_neg)
        else:
            sim = sim.detach().cpu().numpy()
            sim -= min(sim) - 1e-7
            return random_choice(random_index, sim/sim.sum(), k=n_neg)

    def adv(self, loss, model, candiate):
        loss.backward(retain_graph=True)
        return model.embed_item.weight.grad[candiate]

    def next_batch(self, batch, model, k=10, loss=None, eps=None, opt=None, choice="hard"):
        n_neg = self.n_neg

        batch = self.train_np[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items, ratings, candidates, candidates_prime = [], [], [], [], [] 

        # generate negative item
        for row in batch:
            user = int(row[self.col_user])
            users.append(user)
            items.append(int(row[self.col_item]))
            ratings.append(float(row[self.col_rating]>=1))
            length = len(candidates)
            while len(candidates) == length:
                candidate = random.choice(self.item_pool_list)
                if candidate not in self.interact_status[user]:
                    candidates.append(candidate)
            candidate_prime = []
            unhit = 0
            while len(candidate_prime) < k:
                candidate_ = random.choice(self.item_pool_list)
                if candidate not in self.interact_status[user] or (max(torch.abs(model.embed_item.weight[candidate_] - model.embed_item.weight[candidate])) <= eps):
                    candidate_prime.append(candidate_)
                else:
                    unhit += 1
                    if unhit > 10*k: 
                        break
            candidates_prime.append(candidate_prime)

        # obtain adv
        user_ = torch.LongTensor(users)
        item_ = torch.LongTensor(candidates)
        rating_ = torch.Tensor([0.]*len(users))
        if self.device == "cuda":
            user_, item_, rating_ = user_.cuda(), item_.cuda(), rating_.cuda()
        r_ = model.forward(user_, item_)
        adv = self.adv(loss(r_, rating_), model, candidates)

        # adversarial negative sampling
        for a, j_ in zip(adv, candidates_prime):
            better_neg = \
                self.knn(a, j_, model.embed_item.weight[j_], n_neg-1, k=k, eps=eps, choice=choice)
            items.extend(better_neg)
            users.extend([user] * (n_neg-1))
            ratings.extend([0.] * (n_neg-1))

        assert len(items) == len(users)
        
        users, items, ratings = \
            torch.LongTensor(np.array(users)),  torch.LongTensor(np.array(items)), torch.Tensor(np.array(ratings)) 

        if self.device == "cuda":
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        return (users, items, ratings)

class ANS2Dataset(DNSDataset):
    def knn(self, target, user, item, candidates, k=None, eps=float("inf"), t=None, choice="hard"):
        j = candidates[item]
        # sampling
        random_index = np.array(random.sample(self.item_pool-self.interact_status[user], k)) if k is not None else np.array(list(self.item_pool-self.interact_status[user]))
        candidates = candidates[random_index]
        # sphere inside
        if eps != float("inf") or t is not None:
            if t is not None:
                nearset = list(torch.topk(((candidates - j).abs().norm(dim=1)), t, largest=False)[1].squeeze().cpu().numpy())
            else:
                nearset = list(((candidates - j).abs().norm(dim=1) <= eps).nonzero().squeeze().cpu().numpy())
            if len(nearset) == 0: 
                print("not in!")
                return item
            candidates = candidates[nearset]
            random_index = random_index[nearset]
        # actual embedding search
        sim = torch.matmul(candidates, target)
        if choice == "hard":
            return random_index[torch.argmax(sim).item()]
        else:
            p = (1 - (sim.cpu().argsort().float() / (len(sim)-1))) ** len(sim)
            p /= p.sum()
            p = p.cpu().numpy()
            return np.random.choice(random_index, 1, p=p)[0]

    def adv(self, loss, model, candiate):
        loss.backward(self.ones)
        return model.embed_item.weight.grad[candiate]

    def next_batch(self, batch, model, k=None, opt=None, choice="hard"):
        # batch size = 1
        batch = self.train.iloc[batch*self.batch_size:(batch+1)*self.batch_size]
        users, items_positive, items_candidate, items_negative = [], [], [], []

        # generate training data
        for row in batch.itertuples():
            user = int(row[self.col_user+1])
            users.append(user)
            items_positive.append(int(row[self.col_item+1]))
            candidate = random.sample(self.item_pool, 1)[0]
            while candidate in self.interact_status[user]:
                candidate = random.sample(self.item_pool, 1)[0]
            items_candidate.append(candidate)
            items_negative.append(candidate)

        users_ = torch.LongTensor(users)
        items_ = torch.LongTensor(items_candidate)
        if self.device == "cuda":
            users_, items_ = users_.cuda(), items_.cuda()
        r_ = model.forward(users_, items_)
        adv = self.adv(-(1-r_).sigmoid().log(), model, items_candidate)
        items_better_negative = []

        for c, (u, i) in enumerate(zip(users, items_candidate)):
            better_neg = \
                self.knn(adv[c], u, i, model.embed_item.weight, t=k, choice=choice)
            items_better_negative.append(better_neg)
            # if self.device == "cuda":
                # print(r_[c], model.forward(torch.LongTensor([users_[c].item()]).cuda(), torch.LongTensor([better_neg]).cuda()))
        opt.zero_grad()
        
        users, items_positive, items_negative, items_better_negative = \
            torch.LongTensor(users), torch.LongTensor(items_positive), torch.LongTensor(items_negative), torch.LongTensor(items_better_negative)

        if self.device == "cuda":
            users, items_positive, items_negative, items_better_negative = users.cuda(), items_positive.cuda(), items_negative.cuda(), items_better_negative.cuda()

        return (users, items_positive, items_negative, items_better_negative)

class VisualDataset(Dataset):
    def __init__(
        self, 
        train, 
        visual,
        col_user=DEFAULT_USER_COL, 
        col_item=DEFAULT_ITEM_COL, 
        col_rating=DEFAULT_RATING_COL, 
        col_timestamp=DEFAULT_TIMESTAMP_COL, 
        seed=None
    ):
        super().__init__(train, col_user=col_user, col_item=col_item, col_rating=col_rating, col_timestamp=col_timestamp, seed=seed)
        self.visual = visual

    def build(self, n_neg=0, batch_size=32, shuffle=True, threshold=None, device="cuda"):
        import torch
        if n_neg:
            self._negative_sampling(n_neg)

        res = (self.users, self.items, self.ratings)
        res = self._shuffle(res) if shuffle else res

        users, items, ratings = torch.LongTensor(res[0]),  res[1],  torch.Tensor(res[2])

        if threshold is not None:
            ratings = (ratings >= threshold).float()

        if device == "cuda":
            users, ratings = users.cuda(), ratings.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u, r = users[k*batch_size: (k+1)*batch_size], ratings[k*batch_size: (k+1)*batch_size]
            i = items[k*batch_size: (k+1)*batch_size]
            v = torch.Tensor(self.visual[i]).cuda() if device == "cuda" else torch.Tensor(self.visual[i])
            i = torch.LongTensor(i).cuda() if device == "cuda" else torch.LongTensor(i)

            yield u, (i, v), r

class VisualBPRDataset(VisualDataset):
    def _negative_sampling(self, n_neg=1):
        self.users, self.items_positive, self.items_negative = [], [], []

        # generate training data
        for row in self.train.itertuples():
            user = int(row[self.col_user+1])
            self.users.append(user)
            self.items_positive.append(int(row[self.col_item+1]))
            for _ in range(n_neg):
                candidate = random.sample(self.item_pool, 1)[0]
                while candidate in self.interact_status[user]:
                    candidate = random.sample(self.item_pool, 1)[0]
                self.items_negative.append(candidate)

        self.users = np.array(self.users)
        self.items_positive = np.array(self.items_positive)
        self.items_negative = np.array(self.items_negative)
        
    def build(self, n_neg=1, batch_size=32, shuffle=True, device="cuda"):
        import torch

        self._negative_sampling(n_neg)

        res = (self.users, self.items_positive, self.items_negative)
        res = self._shuffle(res) if shuffle else res

        users, items_positive, items_negative = torch.LongTensor(res[0]),  res[1],  res[2]
        if device == "cuda":
            users = users.cuda()

        for k in range(int(np.ceil(len(users) / batch_size))):
            u = users[k*batch_size: (k+1)*batch_size]
            i = items_positive[k*batch_size: (k+1)*batch_size]
            j = items_negative[k*batch_size: (k+1)*batch_size]

            v_i = torch.Tensor(self.visual[i]).cuda() if device == "cuda" else torch.Tensor(self.visual[i])
            v_j = torch.Tensor(self.visual[j]).cuda() if device == "cuda" else torch.Tensor(self.visual[j])

            i = torch.LongTensor(i).cuda() if device == "cuda" else torch.LongTensor(i)
            j = torch.LongTensor(j).cuda() if device == "cuda" else torch.LongTensor(j)

            yield u, (i, v_i), (j, v_j)


class ActiveDataset(object):
    """Dataset class for ReAL Active set"""
    def __init__(
            self,
            test,
            active,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_timestamp=DEFAULT_TIMESTAMP_COL,
            seed=None,
        ):
        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp

        # get test and active set
        self.test = test
        self.active = active
        
        # initialize negative sampling for training
        self._init_active_data()

        # set random seed
        random.seed(seed)

    def _init_active_data(self):
        # cold item
        self.cold_item = np.array(list(set(self.test[self.col_item])))
        # active user
        self.active_user = np.array(list(set(self.test[self.col_user])))
        # query table
        self.table = dict()
        for row in self.active.itertuples():
            self.table[(row[self.col_user+1], row[self.col_item+1])] \
                = row[self.col_rating+1]
            
    def query(self, user, item):
        query = (user, item)
        if query in self.table:
            return self.table[query]
        return 0

class LossDataset(object):
    """ Dataset class for loss information in Phase 1"""
    def __init__(
        self,
        data,
        ratio=0.8,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_loss=DEFAULT_LOSS_COL,
        seed=5583,
    ):
        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_loss = col_loss

        # get test and active set
        self.data = data

        # set random seed
        np.random.seed(5583)

        # split data into two parts
        self.split(ratio)
    
    def split(self, ratio):
        msk = np.random.rand(len(self.data)) < ratio
        self.train = self.data[msk]
        self.test = self.data[~msk]
