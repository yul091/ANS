for k in 8 16 20 30 40
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data yelp-1000 \
        --strategy mf-dns \
        --hidden_dim 16 \
        --epochs 200 \
        --lr 0.1 \
        --batch_size 256 \
        --l2 0.01 \
        --optim Adagrad \
        --choice hard \
        --load_ckpt ckpt/yelp-1000/mf-gans-16/2020-01-13-03-58-05_epoch74_ndcg@50.1240_ndcg@100.1587_ndcg@200.1937_ndcg@500.2324_hr@50.1957_hr@100.3039_hr@200.4428_hr@500.6371_auc0.9103/model.p \
        --k $k
done