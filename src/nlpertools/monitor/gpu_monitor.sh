#!/bin/bash

# 配置参数
THRESHOLD_MEM=30    # 内存使用率阈值(%)，低于此值触发
THRESHOLD_UTL=20    # 利用率阈值(%)，低于此值触发
CHECK_INTERVAL=10   # 检查间隔(秒)
TASK_SCRIPT="./your_task.sh"  # 要执行的任务脚本路径

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到nvidia-smi，无法监控GPU"
    exit 1
fi

# 检查任务脚本是否存在
if [ ! -f "$TASK_SCRIPT" ]; then
    echo "错误: 任务脚本 $TASK_SCRIPT 不存在"
    exit 1
fi

echo "开始监控GPU状态..."
echo "阈值设置 - 内存使用率: <$THRESHOLD_MEM%, 利用率: <$THRESHOLD_UTL%"
echo "检查间隔: $CHECK_INTERVAL秒，任务脚本: $TASK_SCRIPT"

while true; do
    # 获取GPU状态信息（取第一个GPU，可根据需要修改索引）
    gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits -i 0) | tr ',' ' ' |
    
    # 解析数据
    mem_used=$(echo "$gpu_info" | tr ',' ' ' awk '{print $1}')
    mem_total=$(echo "$gpu_info" | awk '{print $2}')
    util=$(echo "$gpu_info" | awk '{print $3}')
    
    # 计算内存使用率(%)
    mem_usage=$((100 * mem_used / mem_total))
    
    # 显示当前状态
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$current_time] GPU状态 - 内存使用率: $mem_usage%, 利用率: $util%"
    
    # 检查是否满足条件
    if [ $mem_usage -lt $THRESHOLD_MEM ] && [ $util -lt $THRESHOLD_UTL ]; then
        echo "GPU状态满足条件，开始执行任务..."
        $TASK_SCRIPT
        echo "任务执行完成，退出监控"
        exit 0
    fi
    
    # 等待下一次检查
    sleep $CHECK_INTERVAL
done