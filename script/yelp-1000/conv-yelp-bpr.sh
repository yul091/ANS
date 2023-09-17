for cc in 0.1
do 
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --data yelp-1000 \
        --strategy conv-bpr \
        --hidden_dim 16 \
        --epochs 20 \
        --lr 0.05 \
        --batch_size 256 \
        --l2 "{'p_u': 0.01, 'q_i': 0.01, 'w': 10, 'c': 0.1}" \
        --optim Adagrad \
        --load_ckpt ckpt/yelp-1000/mf-bpr/2020-01-05-19-59-30_epoch9_ndcg0.1634_hr0.5453_auc0.5387/model.p
done