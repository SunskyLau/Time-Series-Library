#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TFMAE
pretrain_model_path="./checkpoints/pretrain_TFMAE_ETTh1_pretrain_512_TFMAE_ETTh1_ftM_sl336_ll48_pl96_dm64_nh8_el3_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"

# 定义关键参数
seq_len=336
pred_len=720

echo "========================================="
echo "Task: $model_name forecast ETTh1 with pretrained weights $seq_len->$pred_len prediction"
echo "Pretrained model: $pretrain_model_path"
echo "========================================="

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id TFMAE_ETTh1_forecast_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --patch_len 12 \
  --stride 12 \
  --padding 12 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 8 \
  --e_layers 3 \
  --d_layers 2 \
  --dropout 0.1 \
  \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  \
  --train_epochs 30 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --des 'Test_with_pretrain' \
  --itr 1 \
  --lradj 'type1' \
  --patience 5 \
  --pretrained_model_path $pretrain_model_path

echo ""
echo "========================================="
echo "Test completed! Check the results in ./results/"
echo "========================================="
