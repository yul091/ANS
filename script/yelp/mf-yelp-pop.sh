for lr in 0.1
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp \
        --strategy mf-pop \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 0.05 \
        --optim Adagrad 
done