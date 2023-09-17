from common import *

class BPR_MF:
    def __init__(self, p_lambda, lr, max_epoch, dimension, eva_k, batch_size):
        self.p_da = p_lambda
        self.lr = lr
        self.epo = max_epoch
        self.k = dimension
        self.eva_k = eva_k
        self.bs = batch_size
    
    def fit(self, train_data, Xvalid, yvalid, record_epoch):
        self.all_u, self.all_b, self.u_b, self.u2i, self.b2i, self.usernotread = get_embedding(train_data)
        self.w = np.random.normal(0,0.01,(len(self.all_u),self.k))
        self.h = np.random.normal(0,0.01,(len(self.all_b),self.k))
        ones = np.ones((self.k, 1))
        self.test_user_book_rank = test_u_b_rank(Xvalid, yvalid)
        self.hr, self.ndcg, self.auc = [], [], []
        self.record_w, self.record_h = [], []
        
        for epoch in tqdm_notebook(range(self.epo)):
            user_list, item_pos_list, item_neg_list = self.shuffle(train_data) # shuffle
            for i in range(len(user_list)): # length of data divides batch_size
                u_idx_batch, i_idx_batch, j_idx_batch = user_list[i], item_pos_list[i], item_neg_list[i]           
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
            if epoch == record_epoch:
                self.record_w = self.w.copy()
                self.record_h = self.h.copy()
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
            item_neg_batch = [self.b2i[random.choice(self.usernotread[u])] for u in user_input[index[begin:begin+self.bs]]]
                
            user_list.append(user_batch)
            item_pos_list.append(item_batch)
            item_neg_list.append(item_neg_batch)
        return user_list, item_pos_list, item_neg_list