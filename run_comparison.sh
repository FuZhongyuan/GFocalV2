#!/bin/bash

# 确保脚本执行失败时立即退出
set -e

# 帮助信息
show_help() {
    echo "GFL框架训练对比测试运行脚本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -e, --epochs NUM     设置最大训练轮数（默认：1）"
    echo "  -i, --iters NUM      设置最大训练迭代数（而不是轮数）"
    echo "  -o, --output DIR     设置结果输出目录（默认：comparison_results）"
    echo "  -j, --jittor-only    仅运行Jittor框架"
    echo "  -p, --pytorch-only   仅运行PyTorch框架"
    echo "  --jittor-log FILE    指定现有的Jittor训练日志文件（不执行训练）"
    echo "  --pytorch-log FILE   指定现有的PyTorch训练日志文件（不执行训练）"
    echo "  -h, --help           显示此帮助信息"
    echo ""
}

# 默认参数
EPOCHS=12
OUTPUT_DIR="comparison_results"
EXTRA_ARGS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -i|--iters)
            EXTRA_ARGS="$EXTRA_ARGS --iters $2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -j|--jittor-only)
            EXTRA_ARGS="$EXTRA_ARGS --jittor-only"
            shift
            ;;
        -p|--pytorch-only)
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
chmod +x compare_gfl_frameworks.py

echo "开始运行GFL框架训练对比测试..."
echo "参数: 轮数=$EPOCHS, 输出目录=$OUTPUT_DIR $EXTRA_ARGS"

# 运行对比脚本
python compare_gfl_frameworks.py --epochs "$EPOCHS" --output-dir "$OUTPUT_DIR" $EXTRA_ARGS

echo "训练对比测试完成!"
if [ -f "$OUTPUT_DIR/framework_comparison_report.md" ]; then
    echo "报告保存在: $(realpath $OUTPUT_DIR/framework_comparison_report.md)"
fi 