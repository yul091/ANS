for lr in 0.05
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data kaggle \
        --strategy conv-bpr \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 "{'p_u': 0.01, 'q_i': 0.01, 'w': 10, 'c': 1}" \
        --optim Adagrad \
        --load_ckpt ckpt/kaggle/mf-bpr/2019-12-25-08-36-08_epoch76_ndcg0.2489_hr0.4639_auc0.8097_loss0.6719/model.p
done
