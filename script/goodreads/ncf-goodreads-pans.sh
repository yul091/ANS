for lr in 0.005 0.001 0.1 0.05 0.01 
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data goodreads \
        --strategy ncf-pans \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2  0 \
        --optim Adam \
        --choice dynamic
done