for k in 8
do 
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
        python main.py \
            --data yelp-1000 \
            --strategy mf-tans-${k} \
            --hidden_dim 16 \
            --epochs 30 \
            --lr 0.1 \
            --batch_size 256 \
            --l2 0.01 \
            --optim Adagrad \
            --choice hard \
            --k $k \
            --eps 1 
done