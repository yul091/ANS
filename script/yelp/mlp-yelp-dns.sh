for lr in 0.005 0.001 
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp \
        --strategy mlp-dns \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 0 \
        --optim Adam 
done