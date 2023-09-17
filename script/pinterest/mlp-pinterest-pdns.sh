for lr in 1e-3 
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data pinterest \
        --strategy mlp-pdns \
        --hidden_dim 16 \
        --layers [256,128,64,32] \
        --epochs 100 \
        --lr 1e-3 \
        --batch_size 256 \
        --l2 0 \
        # --l2 "{'p_u': 1e-7, 'q_i': 1e-7, 'w_0': 0.0, 'w_1': 1e-6, 'w_2': 1e-6, 'w_3': 1e-5}" \
        --optim Adam 
done
