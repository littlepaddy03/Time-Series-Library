#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer_Regression

python -u run.py \
  --task_name yield_regression \
  --is_training 0 \
  --root_path ./ \
  --data_path dataset/global_yield_dataset/ \
  --model_id Global_Yield_Variable_Test \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 200 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 20 \
  --c_out 1 \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 32 \
  --des 'Variable Length Test' \
  --itr 1 \
  --loss 'MSE' \
  --patch_len 16 \
  --static_feat_dim 66 \
  --head_mlp_dim 128 \
  --regions 'us'
