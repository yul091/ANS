for l2 in 0.001 0.005 0.01 0.05 0.1
do 
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --data yelp-1000 \
        --strategy ncf-bpr \
        --hidden_dim 16 \
        --epochs 1000 \
        --lr 0.005 \
        --batch_size 256 \
        --l2 $l2 \
        --optim Adam
done