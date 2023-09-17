for l2 in 0.001
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp_1000 \
        --strategy mf-bpr \
        --hidden_dim 64 \
        --epochs 10 \
        --lr 0.05 \
        --batch_size 256 \
        --l2 0.01 \
        --optim Adagrad 
done