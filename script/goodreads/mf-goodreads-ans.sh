for k in 4 8 16 32
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data goodreads \
        --strategy mf-gans-${k} \
        --hidden_dim 16 \
        --epochs 20 \
        --lr 0.1  \
        --batch_size 256 \
        --l2 0 \
        --optim Adagrad \
        --choice hard \
        --k $k
done
