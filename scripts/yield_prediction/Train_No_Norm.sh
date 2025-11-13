#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# This script trains a model on US Rice data WITHOUT any normalization
# for comparison purposes. The --use_norm flag is set to 0.

python -u run.py \
  --task_name yield_regression \
  --is_training 1 \
  --root_path ./ \
  --data_path dataset/global_yield_dataset/ \
  --model_id Rice_NoNorm \
  --model PatchTST_Regression \
  --data custom \
  --features S \
  --seq_len 365 \
  --pred_len 1 \
  --enc_in 20 \
  --d_model 256 \
  --n_heads 4 \
  --e_layers 3 \
  --d_ff 512 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --static_feat_dim 66 \
  --head_mlp_dim 128 \
  --des 'Exp_No_Norm' \
  --itr 1 \
  --regions 'us' \
  --crop_name 'Rice' \
  --use_norm 0
