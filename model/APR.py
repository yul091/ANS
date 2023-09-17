from common import *

class APR:
    def __init__(self, p_lambda, p_adv, eps, lr, start_epoch, end_epoch, dimension, eva_k, batch_size):
        self.p_da = p_lambda
        self.p_adv = p_adv
        self.eps = eps
        self.lr = lr
        self.s_epo = start_epoch
        self.e_epo = end_epoch
        self.k = dimension
        self.eva_k = eva_k
        self.bs = batch_size
        
    def l2_normalize(self, arr, axis = 0):
        if (axis == 0):
            return arr / np.linalg.norm(arr, axis = axis)
        else:
            return arr / np.linalg.norm(arr, axis = axis).reshape(-1, 1)
    
    def fit(self, w, h, train_data, Xvalid, yvalid):
        self.all_u, self.all_b, self.u_b, self.u2i, self.b2i, self.usernotread = get_embedding(train_data)
        self.hr, self.ndcg, self.auc = [], [], []
        self.w, self.h = w.copy(), h.copy()
        ones = np.ones((self.k, 1))
        self.test_user_book_rank = test_u_b_rank(Xvalid, yvalid)
        ndcg, hr, auc = evaluation(self.test_user_book_rank, self.w, self.h, self.u2i, self.b2i, self.eva_k)
        self.hr.append(hr)
        self.ndcg.append(ndcg)
        self.auc.append(auc)
        print(f'epoch: {self.s_epo}, HR: {hr}, NDCG: {ndcg}, AUC: {auc}')

        for epoch in tqdm_notebook(range(self.s_epo, self.e_epo)):
            user_list, item_pos_list, item_neg_list = self.shuffle(train_data) # shuffle
            for i in range(len(user_list)): # length of data divides batch_size
                u_idx_batch, i_idx_batch, j_idx_batch = user_list[i], item_pos_list[i], item_neg_list[i]           
                w_u, h_i, h_j = self.w[u_idx_batch].copy(), self.h[i_idx_batch].copy(), self.h[j_idx_batch].copy() 
                x_uij_batch = (w_u * (h_i-h_j)).dot(ones)
                 
                sigmoid_x_batch = 1/(1+np.exp(x_uij_batch))
                tau_u = sigmoid_x_batch*(h_j-h_i)
                tau_i = -sigmoid_x_batch*w_u
                tau_j = sigmoid_x_batch*w_u
                
                # update w_u, h_i, h_j
                w_u_adv = w_u + self.eps * self.l2_normalize(tau_u, axis = 1)
                h_i_adv = h_i + self.eps * self.l2_normalize(tau_i, axis = 1)
                h_j_adv = h_j + self.eps * self.l2_normalize(tau_j, axis = 1)
                x_uij_adv_batch = (w_u_adv * (h_i_adv - h_j_adv)).dot(ones)
                sigmoid_x_adv_batch = 1/(1+np.exp(x_uij_adv_batch))
                
                # update w and h
                self.w[u_idx_batch] -= self.lr*(sigmoid_x_batch*(h_j-h_i) + self.p_da*w_u + self.p_adv*sigmoid_x_adv_batch*(h_j_adv - h_i_adv))
                self.h[i_idx_batch] -= self.lr*(-sigmoid_x_batch*w_u + self.p_da*h_i-self.p_adv*sigmoid_x_adv_batch*w_u_adv)
                self.h[j_idx_batch] -= self.lr*(sigmoid_x_batch*w_u + self.p_da*h_j+self.p_adv*sigmoid_x_adv_batch*w_u_adv)

            ndcg, hr, auc = evaluation(self.test_user_book_rank, self.w, self.h, self.u2i, self.b2i, self.eva_k)
            self.hr.append(hr)
            self.ndcg.append(ndcg)
            self.auc.append(auc)
            print(f'epoch: {epoch+1}, HR: {hr}, NDCG: {ndcg}, AUC: {auc}')
    
    def shuffle(self, dataset):
        user_input = dataset['userID'].values
        item_input_pos = dataset['bookID'].values
        #all_items = set(item_input_pos)
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