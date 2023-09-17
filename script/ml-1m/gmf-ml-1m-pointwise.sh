for lr in 0.001 
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data ml-1m \
        --strategy gmf-pointwise \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 1e-7 \
        --optim Adam
done
