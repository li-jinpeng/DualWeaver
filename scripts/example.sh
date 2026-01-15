#!/bin/bash

# Example run script for DualWeaver on ETTh1

# Settings
model_name="timerxl"
adapter="WeaverMLP"
data_name="etth1"
data_path="dataset/ETT-small/ETTh1.csv"
input_channel=7

# TimerXL defaults
pretrain_model_path="hf_ltm/timer-base-84m"
seq_len=2880
input_token_len=96
output_token_len=96
batch_size=16
accum_steps=$((32 / batch_size))

# Experiment parameters
test_pred_len=96
learning_rate="1e-2"

# Distributed settings (single node, 8 GPU example)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nodes=1
gpu_num=8
port=29500

echo "Starting example experiment..."

torchrun \
    --nnodes=$nodes \
    --node_rank=0 \
    --nproc_per_node=$gpu_num \
    --master_port=$port \
    run.py \
    --data_name "$data_name" \
    --data_path "$data_path" \
    --input_channel $input_channel \
    --adapter "$adapter" \
    --model "$model_name" \
    --seq_len $seq_len \
    --input_token_len $input_token_len \
    --output_token_len $output_token_len \
    --test_pred_len $test_pred_len \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --weight_decay 1e-3 \
    --train_epochs 10 \
    --num_workers 4 \
    --pretrained_model_path "$pretrain_model_path" \
    --accum_steps $accum_steps \
    --ddp \
    --scale \
    --use_amp

echo "Done."
