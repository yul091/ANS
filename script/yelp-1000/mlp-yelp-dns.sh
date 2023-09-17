for k in 8 16 20 30 40
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data yelp-1000 \
        --strategy mlp-dns \
        --hidden_dim 16 \
        --epochs 200 \
        --lr 0.1 \
        --batch_size 256 \
        --l2 0.01 \
        --optim Adagrad \
        --choice hard \
        --k $k
done