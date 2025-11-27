#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./checkpoints" ]; then
    mkdir -p ./checkpoints
fi
if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

# 必须指定 logs_path 变量（推荐带双引号）
logs_path="./logs/run_pretrain_forecast.log"

# 唯一运行标识（便于区分不同次脚本执行）
run_id="$(date '+%Y%m%d%H%M%S')-$$"

# 当前脚本名
script_name="$(basename "$0")"

log() {
    # 构造整行字符串，避免把格式字符串首字符作为 printf 的选项
    printf "%s\n" "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$logs_path"
}

# 错误与退出处理：确保异常退出时也将信息写入日志，且脚本结束时追加结束标记
on_err() {
    local exit_code=$?
    local lineno=${1:-unknown}
    log "ERROR: Command failed with exit code ${exit_code} at line ${lineno}"
}

on_exit() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        log "Script exited abnormally with code ${exit_code}"
    else
        log "Script completed successfully"
    fi

    if [ "$exit_code" -ne 0 ]; then
        log "RUN END run_id=${run_id} script=${script_name} STATUS=FAILED code=${exit_code}"
    else
        log "RUN END run_id=${run_id} script=${script_name} STATUS=OK"
    fi
    # 更明显的结束分隔：使用多行五角星号围绕 RUN END 信息
    printf '%s
' '******************************************************************************' | tee -a "$logs_path"
    printf '%s
' '******************************************************************************' | tee -a "$logs_path"
    printf '%s
' '******************************************************************************' | tee -a "$logs_path"
    printf '%s
' '******************************************************************************' | tee -a "$logs_path"
    # 在 RUN end 之后追加两个空行，便于视觉区分
    printf '%s\n' '' | tee -a "$logs_path"
    printf '%s\n' '' | tee -a "$logs_path"

}

# 捕获错误和信号
trap 'on_err $LINENO' ERR
trap 'on_exit' EXIT
trap 'log "Script interrupted by SIGINT"; exit 130' INT
trap 'log "Script terminated by SIGTERM"; exit 143' TERM

# 写入开始标记到日志（使用多行五角星号作为更明显的分隔符，便于区分不同脚本运行）
printf '%s
' '******************************************************************************' | tee -a "$logs_path"
printf '%s
' '******************************************************************************' | tee -a "$logs_path"
printf '%s
' '******************************************************************************' | tee -a "$logs_path"
printf '%s
' '******************************************************************************' | tee -a "$logs_path"
log "RUN START run_id=${run_id} script=${script_name}"


# 共享变量
root_path=./dataset/ETT-small/
data_path=ETTm2.csv
data=ETTm2
model_name=TFMAE
prtrain_epochs=100
finetune_epochs=30

patch_len=12
stride=12
padding=12

## 模型参数
seq_len=336
pred_lengths=(96 192 336 720) # 预测长度列表
d_model=64
d_ff=128
n_heads=8
e_layers=3
bs=32


# 第一步: 预训练
log "TASK=pretrain STATUS=START run_id=${run_id} model=${model_name} data=${data} seq_len=${seq_len}"

python -u run.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id TFMAE_${data}_pretrain_${seq_len} \
    --model $model_name \
    --data $data \
    --seq_len $seq_len \
    --patch_len $patch_len \
    --stride $stride \
    --padding $padding \
    --d_model $d_model \
    --d_ff $d_ff \
    --n_heads $n_heads \
    --e_layers $e_layers \
    --dropout 0.2 \
    --mask_ratio 0.4 \
    --lambda_t 1.0 \
    --lambda_f 1.0 \
    --train_epochs $prtrain_epochs \
    --batch_size $bs \
    --learning_rate 0.0001 \
    --patience 5 \
    --des 'Exp' \
    --itr 1 \
    --use_gpu True \
    --gpu 0 \
    --lradj 'cosine' \
    --logs_path $logs_path

# 检查预训练是否成功（捕获返回码并记录）
rc=$?
if [ $rc -ne 0 ]; then
    log "TASK=pretrain STATUS=FAILED run_id=${run_id} rc=${rc}"
    exit $rc
fi

# ========================================
# 第二步: 使用预训练模型进行微调和预测
# ========================================

# 定义预训练模型路径：直接从日志文件的最后一行读取（要求最后一行仅为 checkpoint 路径）
if [ ! -f "$logs_path" ]; then
    log "Error: logs file $logs_path not found"
    exit 1
fi

# 尝试从日志中匹配最近一次写入的 checkpoint 路径（形如 ./checkpoints/.../checkpoint.pth）
pretrain_model_path=$(grep -oE './checkpoints/[^[:space:]]+/checkpoint\.pth' "$logs_path" | tail -n 1)

# 回退：若未匹配到，则取日志最后一行（原有逻辑）
if [ -z "$pretrain_model_path" ]; then
    pretrain_model_path=$(awk 'NF{line=$0} END{print line}' "$logs_path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
fi

if [ -z "$pretrain_model_path" ]; then
    log "Error: no pretrained model path found in $logs_path"
    exit 1
fi

if [ ! -f "$pretrain_model_path" ]; then
    log "Error: pretrained model file not found at: $pretrain_model_path"
    exit 1
fi

log "TASK=pretrain STATUS=COMPLETE run_id=${run_id} model_path=${pretrain_model_path}"
# 在任务间插入一个空行，便于不同任务间区分
printf '%s\n' '' | tee -a "$logs_path"


for pred_len in "${pred_lengths[@]}"
do
    # 开始 finetune（前一任务后已有一个空行作为分隔）
    log "TASK=finetune STATUS=START run_id=${run_id} model=${model_name} pred_len=${pred_len} seq_len=${seq_len}"
    log "Using pretrained model: ${pretrain_model_path}"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id TFMAE_${data}_forecast_${seq_len}_${pred_len} \
        --model $model_name \
        --data $data \
        --seq_len $seq_len \
        --patch_len $patch_len \
        --stride $stride \
        --padding $padding \
        --d_model $d_model \
        --d_ff $d_ff \
        --n_heads $n_heads \
        --e_layers $e_layers \
        --dropout 0.1 \
        --label_len 48 \
        --pred_len $pred_len \
        --factor 3 \
        --train_epochs $finetune_epochs \
        --batch_size $bs \
        --learning_rate 0.0001 \
        --des 'Test_with_pretrain' \
        --itr 1 \
        --lradj 'type1' \
        --patience 5 \
        --pretrained_model_path $pretrain_model_path \
        --logs_path $logs_path
    
    rc=$?
    if [ $rc -ne 0 ]; then
        log "TASK=finetune STATUS=FAILED run_id=${run_id} pred_len=${pred_len} rc=${rc}"
        log "Warning: Fine-tuning for pred_len=${pred_len} failed, continuing..."
    else
        log "TASK=finetune STATUS=COMPLETE run_id=${run_id} pred_len=${pred_len}"
        log "Fine-tuning for pred_len=${pred_len} completed successfully!"
    fi
    # 在 finetune 任务之间插入一个空行
    printf '%s\n' '' | tee -a "$logs_path"
done

log "All tasks completed for run_id=${run_id}"
log "Pre-training and fine-tuning finished."
log "Results summary: checkpoints=./checkpoints/ results=./results/ logs=./logs/"


# 结束标记由 on_exit trap 处理，无需在此手动重复输出