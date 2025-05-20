#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}开始GFL框架训练日志分析...${NC}"

# 设置日志路径
JITTOR_LOG="/root/data-fs/GFocalV2/comparison_results/jittor/train_full_output.log"
PYTORCH_LOG="/root/data-fs/GFocalV2/comparison_results/pytorch/train_full_output.log"
OUTPUT_DIR="framework_comparison"

# 安装所需依赖
echo -e "${GREEN}检查并安装必要的Python库...${NC}"
pip install matplotlib numpy pandas seaborn -q

# 运行分析脚本
echo -e "${GREEN}开始运行分析脚本...${NC}"
python analyze_framework_logs.py --jittor-log "$JITTOR_LOG" --pytorch-log "$PYTORCH_LOG" --output-dir "$OUTPUT_DIR"

echo -e "${GREEN}分析完成！报告和可视化结果已保存至 $OUTPUT_DIR 目录下${NC}"
echo -e "${YELLOW}请查看 $OUTPUT_DIR/README.md 获取详细分析报告${NC}" 