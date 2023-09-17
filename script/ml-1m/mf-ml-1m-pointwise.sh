for lr in 0.005 0.001 
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data ml-1m \
        --strategy mf-pointwise \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 1e-7 \
        --optim Adagrad
done
