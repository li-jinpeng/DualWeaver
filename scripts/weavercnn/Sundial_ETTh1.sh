export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
gpu_num=8
nodes=1
port=$((12345 + RANDOM % 10000))
seq_len=2880
model_name=sundial
test_pred_len=96
adapter=WeaverCNN
data_name=etth1
data_path=./dataset/ETT-small/ETTh1.csv
is_training=1
batch_size=8
accum_steps=$((32 / batch_size))
input_token_len=16
output_token_len=720
test_n_sample=10
pretrain_model_path=hf_ltm/sundial-base-128m
learning_rate=1e-4
input_channel=7
output_channel=14

des="${model_name}_${adapter}_${data_name}_${test_pred_len}_${learning_rate}_${output_channel}"
mkdir -p ./logs
log_file="./logs/${des}.log"
> $log_file

torchrun \
        --nnodes=$nodes \
        --node_rank=0 \
        --nproc_per_node=$gpu_num \
        --master_port=$port \
        run.py \
        --data_name $data_name \
        --data_path $data_path \
        --is_training $is_training \
        --adapter $adapter \
        --model_id $des \
        --model $model_name \
        --seq_len $seq_len \
        --input_token_len $input_token_len \
        --output_token_len $output_token_len \
        --test_pred_len $test_pred_len \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --weight_decay 1e-3 \
        --train_epochs 10 \
        --num_workers 20 \
        --des $des \
        --test_n_sample $test_n_sample \
        --scale \
        --pretrained_model_path $pretrain_model_path \
        --use_amp \
        --accum_steps $accum_steps \
        --ddp \
        --input_channel $input_channel \
        --output_channel $output_channel \
        > $log_file 2>&1
