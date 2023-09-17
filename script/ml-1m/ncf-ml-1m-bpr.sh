for lr in 1e-6
do 
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --data ml-1m \
        --strategy ncf-pointwise \
        --hidden_dim 16 \
        --epochs 100 \
        --lr 1e-6 \
        --batch_size 256 \
        --l2 "{'p_u': 1e-6, 'q_i': 1e-6, 'w_0': 1e-6, 'w_1': 1e-6, 'w_2': 1e-6, 'w_3': 1e-6}" \
        --optim SGD \
        --load_gmf_ckpt ckpt/ml-1m/gmf-pointwise/2020-01-15-21-32-06_epoch32_ndcg@50.3513_ndcg@100.4086_ndcg@200.4462_ndcg@500.4730_hr@50.5139_hr@100.6912_hr@200.8396_hr@500.9720_auc0.9020/model.p \
        --load_mlp_ckpt ckpt/ml-1m/mlp-pointwise/2020-01-15-22-22-40_epoch44_ndcg@50.3420_ndcg@100.3980_ndcg@200.4377_ndcg@500.4653_hr@50.4983_hr@100.6725_hr@200.8293_hr@500.9667_auc0.8941/model.p
done