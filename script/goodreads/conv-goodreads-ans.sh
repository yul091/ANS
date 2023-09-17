for lr in 0.05
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data goodreads \
        --strategy conv-ans \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2  "{'p_u': 0.01, 'q_i': 0.01, 'w': 10, 'c': 1}" \
        --optim Adagrad \
        --load_ckpt ckpt/goodreads/mf-bpr-new/2020-01-01-14-44-15_epoch98_ndcg0.5318_hr0.7974_auc0.9363/model.p \
        --choice dynamic 
done