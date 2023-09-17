import numpy as np
import torch

from collections import defaultdict
from concurrent import futures

DEFAULT_USER_COL = 0
DEFAULT_ITEM_COL = 1
DEFAULT_RATING_COL = 2
DEFAULT_TIMESTAMP_COL = 3
DEFAULT_LOSS_COL = 3

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DataSpliter(object):
    @staticmethod
    def random_split(data, ratio):
        """
            data (DataFrame): training data, usually [user, item, rating, timestamp]
            ratio (float): ratio fro training data, rest for testing
        """
        assert ratio < 1 and ratio > 0
        train = data.sample(frac=ratio)
        test = data.iloc[list(set(data.index)-set(train.index))]
        return train.reset_index(drop=True), test.reset_index(drop=True)
    
    @staticmethod
    def cold_start_split(data, user_ratio, item_ratio):
        """ Split as protocol in 'Pairwise Preference Regression for Cold-start Recommendation'
        """
        # split users
        users = data[DEFAULT_USER_COL].unique()
        n = np.arange(len(users))
        np.random.shuffle(n)

        sep = int(user_ratio*len(n))
        existing_user = users[n[:sep]]
        new_user = users[n[sep:]]

        assert len(set(existing_user) & set(new_user))==0
        assert len(existing_user) == sep

        # split items
        items = data[DEFAULT_ITEM_COL].unique()
        n = np.arange(len(items))
        np.random.shuffle(n)

        sep = int(item_ratio*data[DEFAULT_ITEM_COL].nunique())
        existing_item = items[n[:sep]]
        new_item = items[n[sep:]]

        assert len(set(existing_item) & set(new_item))==0
        assert len(existing_item) == sep

        # split dataset
        eu_ei = data.loc[data[DEFAULT_USER_COL].isin(set(existing_user)) & data[DEFAULT_ITEM_COL].isin(set(existing_item))]
        nu_ei = data.loc[data[DEFAULT_USER_COL].isin(set(new_user)) & data[DEFAULT_ITEM_COL].isin(set(existing_item))]
        eu_ni = data.loc[data[DEFAULT_USER_COL].isin(set(existing_user)) & data[DEFAULT_ITEM_COL].isin(set(new_item))]
        nu_ni = data.loc[data[DEFAULT_USER_COL].isin(set(new_user)) & data[DEFAULT_ITEM_COL].isin(set(new_item))]

        return eu_ei.reset_index(drop=True), nu_ei.reset_index(drop=True), \
             eu_ni.reset_index(drop=True), nu_ni.reset_index(drop=True)

    @staticmethod
    def leave_k_out_split(data, k=1, by=DEFAULT_USER_COL, chrono=True, margin=1):
        if chrono == False or len(data.columns.to_list()) <= 3:
            data = data.sample(frac=1)
            df_grouped = data.groupby(by)
        else:
            df_grouped = data.sort_values(DEFAULT_TIMESTAMP_COL).groupby(by)            
        leave_index = []
        for name, group in df_grouped:
            index = group.index
            if len(index) >= (k+margin):
                leave_index.extend(index[-k:])
        test = data.iloc[leave_index]
        train = data.iloc[list(set(data.index)-set(leave_index))] 

        print("leave_k_out split for %d/%d group(s)!" % (len(leave_index)//k, len(df_grouped)))

        return train.reset_index(drop=True), test.reset_index(drop=True)

class Grid:
    def __init__(self, eps, dim, num, embed):
        self.eps = eps
        self.dim = dim
        self.num = num
        self.map = defaultdict(list)
        index = self.indexing(embed[np.random.randint(num)])
        self.map[index] = list(range(self.num))
        self.inv = {i:index for i in range(self.num)}
        self.update_batch(embed, list(range(num)))

    def indexing(self, vec):
        index = str((vec / self.eps).detach().int().tolist())
        return index

    def update(self, vec, i):
        old_index = self.inv[i]
        new_index = self.indexing(vec)
        if new_index != old_index:
            self.map[old_index].remove(i)
            self.map[new_index].append(i)
            self.inv[i] = new_index

    def search(self, vec):
        index = self.indexing(vec)
        return self.map[index]

    def update_batch(self, vec, b):
        index = map(str, (vec[b].detach() / self.eps).int().tolist())
        for k, new_index in enumerate(index):
            i = b[k]
            old_index = self.inv[i]
            if new_index != old_index:
                self.map[old_index].remove(i)
                self.map[new_index].append(i)
                self.inv[i] = new_index

    def search_batch(self, vec):
        index = map(str, (vec / self.eps).detach().int().tolist())
        return [self.map[i] for i in index]

    def search_id(self, b):
        return self.map[self.inv[b]]

    def search_id_batch(self, b):
        return [self.map[self.inv[i]] for i in b]

def _choice(options, p):
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(p):
        cum += p
        if x < cum:
            break
    return options[i]

def random_choice(options, k, p):
    if k == 1:
        return _choice(options, p)
    elif k > 1:
        return [_choice(options, p) for _ in range(k)]
    else:
        raise ValueError

