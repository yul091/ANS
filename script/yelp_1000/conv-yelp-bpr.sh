for cc in 1
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp_1000 \
        --strategy conv-bpr \
        --hidden_dim 64 \
        --epochs 1500 \
        --lr 0.05 \
        --batch_size 512 \
        --l2 "{'p_u': 0.01, 'q_i': 0.01, 'w': 10, 'c': 1}" \
        --optim Adagrad \
        --load_ckpt ckpt/yelp_1000/mf-bpr/2020-01-13-11-28-10_epoch9_ndcg@50.1012_ndcg@100.1323_ndcg@200.1637_ndcg@500.2061_hr@50.1564_hr@100.2533_hr@200.3778_hr@500.5918_auc0.9187/model.p
done