#!/bin/bash

# 内存比较脚本
# 用于比较Jittor和PyTorch框架在训练和测试时的内存占用情况

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== GFocalV2 框架内存使用对比 =====${NC}"
echo "此脚本将比较Jittor和PyTorch在GFocalV2模型训练和测试时的内存占用情况"

# 检查并安装所需的Python包
echo -e "${GREEN}正在检查所需的Python包...${NC}"

pip install psutil pandas matplotlib seaborn --quiet

# 如果需要控制脚本运行时间，可以修改Python脚本中的--max-iters参数
echo -e "${GREEN}准备运行内存比较脚本...${NC}"
echo "注意：为了快速演示，训练过程限制为20次迭代，测试过程限制为100个检测"
echo "如需完整运行，请修改compare_memory_usage.py脚本中相应的参数"

# 创建结果目录
mkdir -p memory_comparison_results

# 运行Python脚本
echo -e "${GREEN}开始执行内存比较...${NC}"
python compare_memory_usage.py

echo -e "${GREEN}内存比较完成！${NC}"
echo "详细结果已保存在memory_comparison_results目录中"
echo "- memory_comparison_summary.csv: 所有测试的内存使用摘要"
echo "- memory_comparison_summary.png: 比较图表"
echo "- [框架]_[模式]_[时间戳].json: 每次运行的详细内存数据"
echo "- [框架]_[模式]_[时间戳].png: 每次运行的内存使用图表"
echo "- [框架]_[模式]_log.txt: 每次运行的输出日志"

# 显示结果摘要
echo -e "${YELLOW}===== 结果摘要 =====${NC}"
if [ -f "memory_comparison_results/memory_comparison_summary.csv" ]; then
    echo "框架内存使用对比："
    cat memory_comparison_results/memory_comparison_summary.csv
else
    echo "未生成结果摘要文件。请检查运行日志查看可能的错误。"
fi 