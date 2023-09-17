for k in 16
do 
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --data goodreads \
        --strategy mf-dns-new \
        --hidden_dim 16 \
        --epochs 1000 \
        --lr 0.05  \
        --batch_size 256 \
        --l2 0 \
        --optim Adagrad \
        --choice hard \
        --k $k
done
