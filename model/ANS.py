from common import *

class BPR_MF_ANS:
    def __init__(self, ada, p_lambda, lr, max_epoch, dimension, eva_k, batch_size):
        self.ada = ada
        self.p_da = p_lambda
        self.lr = lr
        self.epo = max_epoch
        self.k = dimension
        self.eva_k = eva_k
        self.bs = batch_size
        
    def cos_sim(self, a, b):
        dot = a.dot(b)
        return [dot[i]/(np.linalg.norm(a[i])) for i in range(self.ada)]
        
    def adv_cos(self, candidates, j, grad_j):
        can_idx = [self.b2i[can] for can in candidates]
        adv_grad_can = self.h[can_idx]-self.h[self.b2i[j]]
        sim_list = self.cos_sim(adv_grad_can, grad_j)
        j_adv = candidates[np.argmax(sim_list)]
        return j_adv
    
    def calculate_grad(self, u_idx_batch, i_idx_batch, j_idx_batch, ones):
        x_uij_batch = (self.w[u_idx_batch] * (self.h[i_idx_batch]-self.h[j_idx_batch])).dot(ones)
        sigmoid_x_batch = 1/(1+np.exp(x_uij_batch))
        grad_j_batch = sigmoid_x_batch*self.w[u_idx_batch] + self.p_da*self.h[j_idx_batch]
        return grad_j_batch
        
    def fit(self, train_data, Xvalid, yvalid):
        self.all_u, self.all_b, self.u_b, self.u2i, self.b2i, self.usernotread = get_embedding(train_data)
        self.w = np.random.normal(0,0.01,(len(self.all_u),self.k))
        self.h = np.random.normal(0,0.01,(len(self.all_b),self.k))
        self.hr, self.ndcg, self.auc = [], [], []
        ones = np.ones((self.k, 1))
        self.test_user_book_rank = test_u_b_rank(Xvalid, yvalid)

        for epoch in tqdm_notebook(range(self.epo)):
            user_list, item_pos_list, item_neg_list, item_neg_ones = self.shuffle(train_data) # shuffle
            
            for i in range(len(user_list)): # length of data divides batch_size
                u_idx_batch, i_idx_batch, j_idx_one = user_list[i], item_pos_list[i], item_neg_ones[i]       
                candidates = item_neg_list[i] # (batch_size, ada)
                grad_j_batch = self.calculate_grad(u_idx_batch, i_idx_batch, j_idx_one, ones)
                j_idx_adv = [self.adv_cos(candidates[i], j_idx_one[i], grad_j_batch[i]) for i in range(self.bs)]
                # first we update the random chosed j
                x_uij_batch = (self.w[u_idx_batch] * (self.h[i_idx_batch]-self.h[j_idx_one])).dot(ones)
                sigmoid_x_batch = 1/(1+np.exp(x_uij_batch))
                # update w and h
                self.w[u_idx_batch] -= self.lr*(sigmoid_x_batch*(self.h[j_idx_one]-self.h[i_idx_batch]) + self.p_da*self.w[u_idx_batch])
                self.h[i_idx_batch] -= self.lr*(sigmoid_x_batch*(-self.w[u_idx_batch]) + self.p_da*self.h[i_idx_batch])
                self.h[j_idx_one] -= self.lr*(sigmoid_x_batch*self.w[u_idx_batch] + self.p_da*self.h[j_idx_one])
                
                # then we update the corresponding adversarial j_prime
                x_uij_adv = (self.w[u_idx_batch] * (self.h[i_idx_batch]-self.h[j_idx_adv])).dot(ones)
                sigmoid_x_adv = 1/(1+np.exp(x_uij_adv))
                # update w and h
                self.w[u_idx_batch] -= self.lr*(sigmoid_x_adv*(self.h[j_idx_adv]-self.h[i_idx_batch]) + self.p_da*self.w[u_idx_batch])
                self.h[i_idx_batch] -= self.lr*(sigmoid_x_adv*(-self.w[u_idx_batch]) + self.p_da*self.h[i_idx_batch])
                self.h[j_idx_adv] -= self.lr*(sigmoid_x_adv*self.w[u_idx_batch] + self.p_da*self.h[j_idx_adv])
                
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
        user_list, item_pos_list, item_neg_list, item_neg_one = [], [], [], []
        for i in range(num_batch):
            begin = i * self.bs
            user_batch = [self.u2i[u] for u in user_input[index[begin:begin+self.bs]]]
            item_batch = [self.b2i[b] for b in item_input_pos[index[begin:begin+self.bs]]]
            item_neg_batch = [[self.b2i[j] for j in random.sample(self.usernotread[u],self.ada)] for u in user_input[index[begin:begin+self.bs]]]
            item_neg_batch2 = [self.b2i[random.choice(self.usernotread[u])] for u in user_input[index[begin:begin+self.bs]]]
            
            user_list.append(user_batch)
            item_pos_list.append(item_batch)
            item_neg_list.append(item_neg_batch)
            item_neg_one.append(item_neg_batch2)
        return user_list, item_pos_list, item_neg_list, item_neg_one