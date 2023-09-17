for l2 in 0.001 0.005 0.01 0.05 0.1
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data yelp_1000 \
        --strategy ncf-gans \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 0.005 \
        --batch_size 256 \
        --l2 $l2 \
        --optim Adam \
        --choice dynamic
done