#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TFMAE
pretrain_model_path="./checkpoints/pretrain_ETTh1_96_TFMAE_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth"

# ETTh1 预测任务 - 96 预测点
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 2 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --pretrained_model_path $pretrain_model_path

# ETTh1 预测任务 - 192 预测点
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --pretrained_model_path $pretrain_model_path

# ETTh1 预测任务 - 336 预测点
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --pretrained_model_path $pretrain_model_path

# ETTh1 预测任务 - 720 预测点
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 16 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 1 \
  --pretrained_model_path $pretrain_model_path
