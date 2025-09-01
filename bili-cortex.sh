#!/bin/bash

# Bili-Cortex 启动脚本

# 检查虚拟环境
if [[ ! -d ".venv" ]]; then
    echo "错误: 虚拟环境不存在，请先运行 ./setup.sh"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 运行主程序
PYTHONPATH="$(pwd)" python src/main.py "$@"