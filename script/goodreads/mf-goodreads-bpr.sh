for lr in 0.05
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data goodreads \
        --strategy mf-bpr \
        --hidden_dim 16 \
        --epochs 1000 \
        --lr $lr \
        --batch_size 256 \
        --l2 0 \
        --optim Adagrad 
done
