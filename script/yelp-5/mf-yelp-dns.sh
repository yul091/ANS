for k in 2 4 8 16
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data yelp-5 \
        --strategy mf-dns \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 0.005 \
        --batch_size 256 \
        --l2 0.001 \
        --optim Adagrad \
        --eval_workers 8 \
        --choice hard \
        --k $k
done