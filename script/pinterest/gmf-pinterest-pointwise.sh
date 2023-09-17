for lr in 1e-3
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data pinterest \
        --strategy gmf-pointwise \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 1e-3 \
        --batch_size 256 \
        --l2 "{'p_u': 1e-7, 'q_i': 1e-7}" \
        --optim Adam
done
