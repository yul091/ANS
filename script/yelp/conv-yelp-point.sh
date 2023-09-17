for lr in 0.05 0.01 0.005 0.001 0.1 
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data yelp \
        --strategy conv-point \
        --hidden_dim 64 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2  "{'p_u': 0.01, 'q_i': 0.01, 'w': 10, 'c': 1}" \
        --optim Adagrad 
done