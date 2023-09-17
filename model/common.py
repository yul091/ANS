from collections import defaultdict
from tqdm import *

import numpy as np
import random

def test_u_b_rank(Xvalid, yvalid):
    test_user_book_rank = defaultdict(list)
    for i in range(len(Xvalid)):
        u, b = Xvalid[i][0], Xvalid[i][1]
        if yvalid[i] == 0:
            test_user_book_rank[u].append(b)
        else:
            test_user_book_rank[u].insert(0,b)
    return test_user_book_rank

def get_embedding(data):
    users, books = data['userID'].values, data['bookID'].values
    user_books = defaultdict(set)
    usernotread = defaultdict(list)
    for i in range(len(users)):
        user_books[users[i]].add(books[i])
    all_users, all_books = list(set(users)), set(books)
    # embedding
    user2index, book2index = defaultdict(int), defaultdict(int)
    for i in range(len(all_users)):
        u = all_users[i]
        user2index[u] = i
        usernotread[u] = list(all_books-user_books[u])
    all_books = list(all_books)
    for j in range(len(all_books)):
        book2index[all_books[j]] = j
        
    return all_users, all_books, user_books, user2index, book2index, usernotread

def evaluation(test_user_book_rank, w, h, u2i, b2i, K):
    predict_user_book = defaultdict(list)
    for u in test_user_book_rank.keys():    
        for b in test_user_book_rank[u]:
            predict_user_book[u].append(w[u2i[u]].dot(h[b2i[b]].T))
        predict_user_book[u] = np.array(predict_user_book[u])

    ndcg, hr, auc = 0, 0, 0
    for u in test_user_book_rank.keys():
        position = sum(predict_user_book[u][1:]>predict_user_book[u][0])
        ndcg += np.log(2) / np.log(position + 2) if position < K else 0
        hr += int(position < K)
        auc += 1 - (position / (len(predict_user_book[u])-1))

    ndcg = ndcg/len(test_user_book_rank.keys())
    hr = hr/len(test_user_book_rank.keys())
    auc = auc/len(test_user_book_rank.keys())
    return ndcg, hr, auc