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
    echo "  -e, --epochs NUM     设置最大训练轮数（默认：12）"
    echo "  -i, --iters NUM      设置最大训练迭代数（而不是轮数）"
    echo "  -o, --output DIR     设置结果输出目录（默认：comparison_results）"
    echo "  -j, --jittor-only    仅运行Jittor框架"
    echo "  -p, --pytorch-only   仅运行PyTorch框架"
    echo "  --no-train           跳过训练，仅使用已有日志进行对比分析"
    echo "  --jittor-log FILE    指定现有的Jittor训练日志文件（配合--no-train使用）"
    echo "  --pytorch-log FILE   指定现有的PyTorch训练日志文件（配合--no-train使用）"
    echo "  -h, --help           显示此帮助信息"
    echo ""
}

# 默认参数
EPOCHS=12
OUTPUT_DIR="comparison_results"
RUN_JITTOR=true
RUN_PYTORCH=true
NO_TRAIN=false
JITTOR_LOG=""
PYTORCH_LOG=""
ITERS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -i|--iters)
            ITERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -j|--jittor-only)
            RUN_JITTOR=true
            RUN_PYTORCH=false
            shift
            ;;
        -p|--pytorch-only)
            RUN_JITTOR=false
            RUN_PYTORCH=true
            shift
            ;;
        --no-train)
            NO_TRAIN=true
            shift
            ;;
        --jittor-log)
            JITTOR_LOG="$2"
            shift 2
            ;;
        --pytorch-log)
            PYTORCH_LOG="$2"
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

# 确保对比脚本可执行
chmod +x compare_gfl_frameworks.py

echo "开始运行GFL框架训练对比测试..."
echo "参数: 轮数=$EPOCHS, 输出目录=$OUTPUT_DIR"
echo "Jittor训练: $([ "$RUN_JITTOR" = true ] && echo "是" || echo "否"), PyTorch训练: $([ "$RUN_PYTORCH" = true ] && echo "是" || echo "否")"

# 创建输出目录结构
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/jittor"
mkdir -p "$OUTPUT_DIR/pytorch"
mkdir -p "$OUTPUT_DIR/report"

# 如果需要训练
if [ "$NO_TRAIN" = false ]; then
    # 运行Jittor训练
    if [ "$RUN_JITTOR" = true ]; then
        JITTOR_LOG="$OUTPUT_DIR/jittor/train_full_output.log"
        echo "开始运行Jittor训练，捕获所有输出到: $JITTOR_LOG"
        
        # 创建工作目录
        mkdir -p "$OUTPUT_DIR/jittor/work_dirs"
        
        # 运行命令并捕获所有输出
        (cd /root/data-fs/GFocalV2/GFocalV2Jittor && \
         pip install -e . && \
         python tools/train.py configs/gfl/gfl_r50_fpn_coco_1x_enhanced.yml \
         --work-dir "$OUTPUT_DIR/jittor/work_dirs" \
         $([ ! -z "$ITERS" ] && echo "--iters $ITERS")) 2>&1 | tee "$JITTOR_LOG"
        
        echo "Jittor训练完成，输出已保存到: $JITTOR_LOG"
    fi
    
    # 运行PyTorch训练
    if [ "$RUN_PYTORCH" = true ]; then
        PYTORCH_LOG="$OUTPUT_DIR/pytorch/train_full_output.log"
        echo "开始运行PyTorch训练，捕获所有输出到: $PYTORCH_LOG"
        
        # 创建工作目录
        mkdir -p "$OUTPUT_DIR/pytorch/work_dirs"
        
        # 运行命令并捕获所有输出
        (cd /root/data-fs/GFocalV2/GFocalV2Pytorch && \
         pip install -e . && \
         python tools/train.py configs/gfl/gfl_r50_fpn_1x_coco.py \
         --work-dir "$OUTPUT_DIR/pytorch/work_dirs" \
         $([ ! -z "$ITERS" ] && echo "--max-iters $ITERS")) 2>&1 | tee "$PYTORCH_LOG"
        
        echo "PyTorch训练完成，输出已保存到: $PYTORCH_LOG"
    fi
else
    echo "跳过训练，直接使用提供的日志文件进行分析..."
    # 检查日志文件
    if [ "$RUN_JITTOR" = true ] && [ -z "$JITTOR_LOG" ]; then
        echo "错误: 启用了Jittor分析但没有提供日志文件。请使用--jittor-log指定日志文件。"
        exit 1
    fi
    
    if [ "$RUN_PYTORCH" = true ] && [ -z "$PYTORCH_LOG" ]; then
        echo "错误: 启用了PyTorch分析但没有提供日志文件。请使用--pytorch-log指定日志文件。"
        exit 1
    fi
fi

# 构建分析命令的参数
ANALYSIS_ARGS=""
[ "$RUN_JITTOR" = true ] && [ ! -z "$JITTOR_LOG" ] && ANALYSIS_ARGS="$ANALYSIS_ARGS --jittor-log $JITTOR_LOG"
[ "$RUN_PYTORCH" = true ] && [ ! -z "$PYTORCH_LOG" ] && ANALYSIS_ARGS="$ANALYSIS_ARGS --pytorch-log $PYTORCH_LOG"
[ "$RUN_JITTOR" = false ] && ANALYSIS_ARGS="$ANALYSIS_ARGS --pytorch-only"
[ "$RUN_PYTORCH" = false ] && ANALYSIS_ARGS="$ANALYSIS_ARGS --jittor-only"

# 运行对比分析
echo "运行对比分析，输出目录: $OUTPUT_DIR/report"
python compare_gfl_frameworks.py --epochs "$EPOCHS" --output-dir "$OUTPUT_DIR/report" $ANALYSIS_ARGS

echo "训练对比测试完成!"
if [ -f "$OUTPUT_DIR/report/framework_comparison_report.md" ]; then
    echo "报告保存在: $(realpath $OUTPUT_DIR/report/framework_comparison_report.md)"
fi 