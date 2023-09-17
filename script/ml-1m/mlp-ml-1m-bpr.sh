for lr in 1e-3 
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data ml-1m \
        --strategy mlp-pointwise \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 1e-3 \
        --batch_size 256 \
        --l2 "{'p_u': 1e-6, 'q_i': 1e-6, 'w_0': 1e-6, 'w_1': 1e-6, 'w_2': 1e-6, 'w_3': 1e-6}" \
        --optim Adam
done
