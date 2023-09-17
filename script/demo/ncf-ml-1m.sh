CUDA_VISIBLE_DEVICES=1 python main.py \
    --data ml-1m \
    --strategy ncf-pointwise \
    --hidden_dim 8 \
    --epochs 20 \
    --lr 0.005 \
    --batch_size 256 \
    --l2 0 \
    --optim Adam 
