for lr in 1e-7
do 
    CUDA_VISIBLE_DEVICES=3 /home/yufei/anaconda3/bin/python main.py \
        --data pinterest \
        --strategy ncf-pointwise \
        --hidden_dim 16 \
        --layers [256,128,64,32] \
        --epochs 100 \
        --lr 1e-7 \
        --batch_size 256 \
        --l2 "{'p_u': 1e-7, 'q_i': 1e-7, 'w_0': 1e-6, 'w_1': 1e-6, 'w_2': 1e-5}" \
        --optim SGD \
	    --load_mlp_ckpt ckpt/pinterest/mlp-pointwise/2020-01-16-23-51-24_epoch13_ndcg@50.4843_ndcg@100.5413_ndcg@200.5658_ndcg@500.5728_hr@50.6883_hr@100.8630_hr@200.9583_hr@500.9921_auc0.9516/model.p \
        --load_gmf_ckpt ckpt/pinterest/gmf-pointwise/2020-01-16-01-03-10_epoch96_ndcg@50.4959_ndcg@100.5508_ndcg@200.5736_ndcg@500.5802_hr@50.7017_hr@100.8700_hr@200.9586_hr@500.9906_auc0.9520/model.p 
done
