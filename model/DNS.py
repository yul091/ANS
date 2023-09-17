from common import *

class BPR_MF_DNS:
    def __init__(self, beta, ada, p_lambda, lr, max_epoch, dimension, eva_k, batch_size):
        self.beta = beta
        self.ada = ada
        self.p_da = p_lambda
        self.lr = lr
        self.epo = max_epoch
        self.k = dimension
        self.eva_k = eva_k
        self.bs = batch_size
        
    def DNS(self, neg_idx, u_idx):
        j_score_list = list(self.w[u_idx].dot(self.h[neg_idx].T)) # (1, ada)
        j_score_sort = sorted(j_score_list, reverse=True)
        # create linear probabilities associated with score ranking
        neg_p = [1-(1-self.beta)*i/len(j_score_sort) for i in range(self.ada)]
        p_ = [p/sum(neg_p) for p in neg_p]
        j_score = np.random.choice(j_score_sort, p=p_)
        j_idx = neg_idx[j_score_list.index(j_score)]
        return j_idx
    
    def fit(self, train_data, Xvalid, yvalid):
        self.all_u, self.all_b, self.u_b, self.u2i, self.b2i, self.usernotread = get_embedding(train_data)
        self.w = np.random.normal(0,0.01,(len(self.all_u),self.k))
        self.h = np.random.normal(0,0.01,(len(self.all_b),self.k))
        self.hr, self.ndcg, self.auc = [], [], []
        ones = np.ones((self.k, 1))
        self.test_user_book_rank = test_u_b_rank(Xvalid, yvalid)

        for epoch in tqdm_notebook(range(self.epo)):
            user_list, item_pos_list, item_neg_list = self.shuffle(train_data) # shuffle        
            for i in range(len(user_list)): # length of data divides batch_size
                u_idx_batch, i_idx_batch = user_list[i], item_pos_list[i]         
                neg_idx_batch = item_neg_list[i] # (batch_size, ada)
                j_idx_batch = [self.DNS(neg_idx_batch[i], u_idx_batch[i]) for i in range(self.bs)]
                
                x_uij_batch = (self.w[u_idx_batch] * (self.h[i_idx_batch]-self.h[j_idx_batch])).dot(ones)
                sigmoid_x_batch = 1/(1+np.exp(x_uij_batch))
                # update w and h
                self.w[u_idx_batch] -= self.lr*(sigmoid_x_batch*(self.h[j_idx_batch]-self.h[i_idx_batch]) + self.p_da*self.w[u_idx_batch])
                self.h[i_idx_batch] -= self.lr*(sigmoid_x_batch*(-self.w[u_idx_batch]) + self.p_da*self.h[i_idx_batch])
                self.h[j_idx_batch] -= self.lr*(sigmoid_x_batch*self.w[u_idx_batch] + self.p_da*self.h[j_idx_batch])

            ndcg, hr, auc = evaluation(self.test_user_book_rank, self.w, self.h, self.u2i, self.b2i, self.eva_k)
            self.hr.append(hr)
            self.ndcg.append(ndcg)
            self.auc.append(auc)
            print(f'epoch: {epoch+1}, HR: {hr}, NDCG: {ndcg}, AUC: {auc}')
      
    def shuffle(self, dataset):
        user_input = dataset['userID'].values
        item_input_pos = dataset['bookID'].values
        index = list(range(len(user_input)))
        np.random.shuffle(index)
        num_batch = len(user_input) // self.bs
        user_list, item_pos_list, item_neg_list = [], [], []
        for i in range(num_batch):
            begin = i * self.bs                
            user_batch = [self.u2i[u] for u in user_input[index[begin:begin+self.bs]]]
            item_batch = [self.b2i[b] for b in item_input_pos[index[begin:begin+self.bs]]]
            item_neg_batch = [[self.b2i[j] for j in random.sample(self.usernotread[u],self.ada)] for u in user_input[index[begin:begin+self.bs]]]
                
            user_list.append(user_batch)
            item_pos_list.append(item_batch)
            item_neg_list.append(item_neg_batch)
        return user_list, item_pos_list, item_neg_list