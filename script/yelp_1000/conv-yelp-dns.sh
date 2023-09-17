for lr in 0.05
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp_1000 \
        --strategy conv-dns \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 "{'p_u': 0.01, 'q_i': 0.01, 'w': 10, 'c': 0.1}" \
        --optim Adagrad \
        --load_ckpt ckpt/yelp-1000/mf-bpr/2020-01-05-19-59-30_epoch9_ndcg0.1634_hr0.5453_auc0.5387/model.p
done