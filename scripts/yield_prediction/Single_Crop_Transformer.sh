#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name yield_regression \
  --is_training 1 \
  --root_path ./ \
  --data_path dataset/global_yield_dataset/ \
  --model_id Single_Crop_Maize_PatchTST \
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
  --des 'Exp_Single_Crop' \
  --itr 1 \
  --regions 'us,ar,br,cn,in,eu' \
  --crop_name 'Maize'
