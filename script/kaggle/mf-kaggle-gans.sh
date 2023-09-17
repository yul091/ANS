for lr in 0.005 0.001 0.1 0.05 0.01 
do 
    python main.py \
        --data kaggle \
        --strategy ncf-gans-sgd \
        --hidden_dim 16 \
        --epochs 100 \
        --lr $lr \
        --batch_size 64 \
        --l2 0 \
        --optim Adam \
        --device cpu \
        --choice dynamic
done
