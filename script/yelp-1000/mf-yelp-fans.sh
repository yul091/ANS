for k in 2
do 
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
        python main.py \
            --data yelp-1000 \
            --strategy mf-fans-${k} \
            --hidden_dim 16 \
            --epochs 30 \
            --lr 0.1 \
            --batch_size 256 \
            --l2 "{'p_u': 0.01, 'q_i': 0.01}" \
            --optim Adagrad \
            --choice hard \
            --k $k \
            --eps 0.5 \
            --load_ckpt ckpt/yelp-1000/mf-gans-16/2020-01-13-03-58-05_epoch74_ndcg@50.1240_ndcg@100.1587_ndcg@200.1937_ndcg@500.2324_hr@50.1957_hr@100.3039_hr@200.4428_hr@500.6371_auc0.9103/model.p
done