for reg in 0.005 0.01 0.05 0.1
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data ml-100k \
        --strategy mlp-dns-soft \
        --hidden_dim 8 \
        --epochs 20 \
        --lr 0.1 \
        --batch_size 256 \
        --choice soft \
        --l2 "{'p_u': ${reg}, 'q_i': ${reg}}" \
        --optim Adagrad
done
