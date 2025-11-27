if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./checkpoints" ]; then
    mkdir -p ./checkpoints
fi

python -u run.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id TFMAE_ETTh1_pretrain_512 \
    --model TFMAE \
    --data ETTh1 \
    --seq_len 336 \
    --patch_len 12 \
    --stride 12 \
    --padding 12 \
    --d_model 64 \
    --d_ff 128 \
    --n_heads  8 \
    --e_layers 3 \
    --dropout 0.2 \
    --mask_ratio 0.4 \
    --lambda_t 1.0 \
    --lambda_f 2.0 \
    --train_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --patience 5 \
    --des 'Exp' \
    --itr 1 \
    --use_gpu True \
    --gpu 0 \
    --lradj 'cosine'