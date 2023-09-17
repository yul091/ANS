for lr in 0.1 0.05 0.01 0.005 0.001 
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data ml-1m \
        --strategy mf-ans \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 0 \
        --optim Adagrad \
        --choice dynamic
done
