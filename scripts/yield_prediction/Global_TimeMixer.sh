export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

# a100
# d_model=16
# d_ff=32
# batch_size=128

# 3090
d_model=16
d_ff=32
batch_size=128

python -u run.py \
  --task_name yield_regression \
  --is_training 1 \
  --root_path ./ \
  --data_path dataset/global_yield_dataset/ \
  --model_id Global_Yield_$model_name \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 365 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 20 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.01 \
  --loss 'MSE' \
  --patience 5 \
  --train_epochs 30 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --decomp_method 'moving_avg' \
  --channel_independence 0 \
  --static_feat_dim 66 \
  --head_mlp_dim 128 \
  --regions 'us,ar,br,cn,in,eu'
