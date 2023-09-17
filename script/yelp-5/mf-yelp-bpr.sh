for l2 in 0.001
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data yelp-5 \
        --strategy mf-bpr \
        --hidden_dim 16 \
        --epochs 1000 \
        --lr 0.005 \
        --batch_size 256 \
        --l2 0.001 \
        --optim Adagrad \
        --eval_workers 8
done