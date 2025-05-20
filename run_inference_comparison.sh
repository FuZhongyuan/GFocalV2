#!/bin/bash

# 确保脚本执行失败时立即退出
set -e

# 帮助信息
show_help() {
    echo "GFL框架推理性能对比测试运行脚本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -j, --jittor-checkpoint PATH    指定Jittor框架的检查点文件路径"
    echo "  -p, --pytorch-checkpoint PATH   指定PyTorch框架的检查点文件路径"
    echo "  -b, --batch-size NUM           设置推理批次大小（默认：1）"
    echo "  -n, --num-samples NUM          限制测试样本数量"
    echo "  -o, --output DIR               设置结果输出目录（默认：inference_comparison_results）"
    echo "  --jittor-only                  仅测试Jittor框架"
    echo "  --pytorch-only                 仅测试PyTorch框架"
    echo "  --jittor-log FILE              指定现有的Jittor推理日志文件（不执行推理）"
    echo "  --pytorch-log FILE             指定现有的PyTorch推理日志文件（不执行推理）"
    echo "  -h, --help                     显示此帮助信息"
    echo ""
}

# 默认参数
OUTPUT_DIR="inference_comparison_results"
BATCH_SIZE=1
EXTRA_ARGS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jittor-checkpoint)
            EXTRA_ARGS="$EXTRA_ARGS --jittor-checkpoint $2"
            shift 2
            ;;
        -p|--pytorch-checkpoint)
            EXTRA_ARGS="$EXTRA_ARGS --pytorch-checkpoint $2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -n|--num-samples)
            EXTRA_ARGS="$EXTRA_ARGS --num-samples $2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --jittor-only)
            EXTRA_ARGS="$EXTRA_ARGS --jittor-only"
            shift
            ;;
        --pytorch-only)
            EXTRA_ARGS="$EXTRA_ARGS --pytorch-only"
            shift
            ;;
        --jittor-log)
            EXTRA_ARGS="$EXTRA_ARGS --jittor-log $2"
            shift 2
            ;;
        --pytorch-log)
            EXTRA_ARGS="$EXTRA_ARGS --pytorch-log $2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 安装必要的依赖
echo "检查并安装必要的Python依赖..."
pip install matplotlib pandas numpy --quiet

# 确保脚本可执行
chmod +x compare_gfl_inference.py

echo "开始运行GFL框架推理性能对比测试..."
echo "参数: 批次大小=$BATCH_SIZE, 输出目录=$OUTPUT_DIR $EXTRA_ARGS"

# 运行对比脚本
python compare_gfl_inference.py --batch-size "$BATCH_SIZE" --output-dir "$OUTPUT_DIR" $EXTRA_ARGS

echo "推理对比测试完成!"
if [ -f "$OUTPUT_DIR/inference_comparison_report.md" ]; then
    echo "报告保存在: $(realpath $OUTPUT_DIR/inference_comparison_report.md)"
fi