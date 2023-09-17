for reg in 0 0.00001 0.001 0.005 0.001 0.01 0.05 0.1
do 
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --data kaggle \
        --strategy mlp-pointwise \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 0.05 \
        --batch_size 64 \
        --l2 $reg \
        --optim Adagrad
done
