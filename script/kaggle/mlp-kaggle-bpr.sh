for lr in 0.001 0.005 0.01 0.05 0.1
do 
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --data kaggle \
        --strategy mlp-bpr \
        --hidden_dim 8 \
        --epochs 100 \
        --lr $lr \
        --batch_size 256 \
        --l2 0 \
        --optim Adagrad
done
