# CUDA_VISIBLE_DEVICES=3 kernprof -l -v main.py \
for lr in 0.1
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp-1000 \
        --strategy mf-pop \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 0.01 \
        --optim Adagrad 
done