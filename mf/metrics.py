
import numpy as np
import collections 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def metrics(data, requests=['mse']):
    """
        data (array): [u, i, r, r_, loss]
    """
    data = np.array(data)
    res = []
    for r in requests:
        r = r.lower()
        if r == "loss":
            res.append(np.mean(data[ -1]))
        elif r == "mse":
            res.append(mse(*data[[2,3]]))
        elif r == "mae":
            res.append(mae(*data[[2,3]]))
        elif r == "rmse":
            res.append(np.sqrt(mse(*data[[2,3]])))
        elif "ndcg" in r:
            k = 10 if len(r.split("@")) != 2 else int(r.split("@")[1])
            res.append(ndcg_at_k(data[:4], k))
        elif "hr" in r:
            k = 10 if len(r.split("@")) != 2 else int(r.split("@")[1])
            res.append(hr_at_k(data[:4], k))
        elif "precision" in r:
            k = 10 if len(r.split("@")) != 2 else int(r.split("@")[1])
            res.append(precision_at_k(data[:4], k))
        elif "recall" in r:
            k = 10 if len(r.split("@")) != 2 else int(r.split("@")[1])
            res.append(recall_at_k(data[:4], k))
        elif r == "auc":
            res.append(auc(data[:4]))
    print("total measure")
    return requests, res


def ndcg_at_k(x, k=10):
    # binary ratings
    user_true_pred = collections.defaultdict(list)
    for u, i, r, r_ in zip(*x):
        user_true_pred[u].append((r, r_))
    ndcg = dict()
    for u, true_pred_pairs in user_true_pred.items():
        # Sort user ratings by estimated value
        user_true_ratings = sorted(true_pred_pairs, key=lambda x: x[0], reverse=True)
        user_pred_ratings = sorted(true_pred_pairs, key=lambda x: x[1], reverse=True)
        # Number of recommended items in top k
        rec_k = [r for (r, r_) in user_pred_ratings[:k]]
        # Number of relevant items in top k
        n_all_k = sum(r for (r, r_) in user_true_ratings[:k])
        # Avoid divide zero
        if n_all_k == 0: 
            ndcg[u] = 1
            continue
        # IDCG
        idcg = sum((1/np.log2(1+(i+1))) for i in range(int(n_all_k)))
        # DCG
        dcg = sum((1/np.log2(1+(i+1))) for i, r in enumerate(rec_k) if r)
        # NDCG@K
        ndcg[u] =  dcg / idcg
    # Average
    return sum(ndcg.values()) / len(ndcg)

def hr_at_k(x, k=10):
    # binary ratings
    user_true_pred = collections.defaultdict(list)
    for u, i, r, r_ in zip(*x):
        user_true_pred[u].append((r, r_))
    hrs = dict()
    for u, true_pred_pairs in user_true_pred.items():
        # Sort user ratings by estimated value
        user_true_ratings = sorted(true_pred_pairs, key=lambda x: x[0], reverse=True)
        user_pred_ratings = sorted(true_pred_pairs, key=lambda x: x[1], reverse=True)
        # Number of recommended items in top k
        rec_k = [r for (r, r_) in user_pred_ratings[:k]]
        # Number of relevant items in top k
        n_all_k = sum(r for (r, r_) in user_true_ratings[:k])
        # Avoid divide zero
        if n_all_k == 0: 
            print("null: ", u)
            print(user_true_ratings[:k])
            hrs[u] = 1
            continue
        # Hits@K
        hrs[u] = sum(rec_k) / n_all_k 
    # Average
    return sum(hrs.values()) / len(hrs)

def precision_at_k(x, k=10, threshold=0.5):
    # binary ratings
    user_true_pred = collections.defaultdict(list)
    for u, i, r, r_ in zip(*x):
        user_true_pred[u].append((r, r_))

    precisions = dict()
    for u, true_pred_pairs in user_true_pred.items():

        # Sort user ratings by estimated value
        true_pred_pairs.sort(key=lambda x: x[1], reverse=True)

        # Number of relevant items
        n_rel = sum(r for (r, _) in true_pred_pairs)

        # Number of recommended items in top k
        n_rec_k = sum((r_ >= threshold) for (_, r_) in true_pred_pairs[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum((r and (r_ >= threshold))
                              for (r, r_) in true_pred_pairs[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[u] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

    return sum(precisions.values()) / len(precisions)

def recall_at_k(x, k=10, threshold=0.5):
    # binary ratings
    user_true_pred = collections.defaultdict(list)
    for u, i, r, r_ in zip(*x):
        user_true_pred[u].append((r, r_))

    recalls = dict()
    for u, true_pred_pairs in user_true_pred.items():

        # Sort user ratings by estimated value
        true_pred_pairs.sort(key=lambda x: x[1], reverse=True)

        # Number of relevant items
        n_rel = sum(r for (r, _) in true_pred_pairs)

        # Number of recommended items in top k
        n_rec_k = sum((r_ >= threshold) for (_, r_) in true_pred_pairs[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum((r and (r_ >= threshold))
                              for (r, r_) in true_pred_pairs[:k])

        # Recall@K: Proportion of relevant items that are recommended
        recalls[u] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        
    return sum(recalls.values()) / len(recalls)

def auc(x):
    # binary ratings
    user_true_pred = collections.defaultdict(list)
    for u, i, r, r_ in zip(*x):
        user_true_pred[u].append([r, r_])

    aucs = dict()
    for u, true_pred_pairs in user_true_pred.items():
        true_pred_pairs = np.array(true_pred_pairs)
        r, r_ = true_pred_pairs[:, 0], true_pred_pairs[:, 1]
        aucs[u] = roc_auc_score(r, r_) if len(set(r)) == 2 else 1
    
    return sum(aucs.values()) / len(aucs)

def metrics_fast(data, k=[5, 10, 20, 50]):
    # position = sum(data[1:]>data[0]) 
    position = (sum(data[1:]>data[0]) + sum(data[1:]>=data[0])) / 2
    ndcg = [np.log(2) / np.log(position + 2) if position < k_ else 0 for k_ in k]
    hr = [int(position < k_) for k_ in k]
    auc = [1 - (position / (len(data)-1))]
    return ndcg + hr + auc + [position]



        

