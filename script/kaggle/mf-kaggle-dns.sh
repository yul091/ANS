for lr in 0.1 0.05 0.01 0.005 0.001 
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data kaggle \
        --strategy ncf-ans-sgd \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 0.1 \
        --batch_size 64 \
        --l2 0.05 \
        --optim SGD \
        --device cpu \
        --choice dynamic
done
