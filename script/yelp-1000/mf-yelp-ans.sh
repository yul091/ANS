for k in 16 32
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data yelp-1000 \
        --strategy mf-gans-${k} \
        --hidden_dim 16 \
        --epochs 80 \
        --lr 0.1 \
        --batch_size 256 \
        --l2 0.01 \
        --optim Adagrad \
        --choice hard \
        --k $k \
        --eps 100
done