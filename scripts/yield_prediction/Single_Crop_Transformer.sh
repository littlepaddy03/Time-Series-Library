#!/bin/bash

# This script is a template for running single-crop training experiments.
# To train for a different crop, change the --crop_name argument.
# Valid crop names are: Maize, Rice, Soybean, Wheat

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name yield_regression \
  --is_training 1 \
  --root_path ./ \
  --data_path dataset/global_yield_dataset/ \
  --model_id Single_Crop_Yield \
  --model PatchTST_Regression \
  --data custom \
  --features S \
  --seq_len 365 \
  --pred_len 1 \
  --enc_in 20 \
  --d_model 256 \
  --n_heads 4 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --d_ff 512 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --des 'Single Crop Training' \
  --loss 'MSE' \
  --lradj 'type1' \
  --static_feat_dim 66 \
  --regions '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16' \
  --crop_name 'Maize' # <-- CHANGE THIS TO TRAIN FOR A DIFFERENT CROP
