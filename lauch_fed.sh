#!/bin/bash
# 设置日志文件路径
LOG_DIR="mylogs"
mkdir -p $LOG_DIR  # 确保日志目录存在
LOG_FILE="$LOG_DIR/run_$(date +"%Y%m%d_%H%M%S").log"

# 运行 Python 代码并将日志写入文件
#python xxx.py > "$LOG_FILE" 2>&1

# Set PYTHONPATH
export PYTHONPATH=/home/FedASTW:$PYTHONPATH

python ./system/fedastw.py   -num_clients 100 \
                    -com_round 10 \
                    -sample_ratio 0.1 \
                    -sampler_name "uniform" \
                    -all_clients 1 \
                    -batch_size 64 \
                    -epochs 3 \
                    -lr 0.01 \
                    -glr 1 \
                    -dseed 37  \
                    -seed 42  \
                    -dataset cifar10 \
                    -partition dirichlet \
                    -dir 0.1 \
                    -preprocess 1 \
                    -mode 1 \
                    -thresh 0.0 \
                    -a 5 \
                    -b 15\
                    -tw \
| tee -a "$LOG_FILE"
                    # -startup 1 \

# 打印日志路径
echo "Log saved to $LOG_FILE"