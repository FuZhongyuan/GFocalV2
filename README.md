# GFL框架对比测试工具

这个工具用于自动化对比PyTorch和Jittor框架下的GFL（Generalized Focal Loss）模型实现的训练效果和性能。

## 功能介绍

该工具可以：

1. 分别在PyTorch和Jittor框架下运行GFL模型的训练
2. 收集训练过程中的日志数据
3. 分析并对比两个框架下的训练损失、性能和结果
4. 生成对比报告和可视化图表

## 文件说明

- `compare_gfl_frameworks.py`: 主要的Python脚本，负责运行训练、解析日志和生成报告
- `run_comparison.sh`: 简化运行测试的Shell脚本
- `README.md`: 本说明文档

## 使用方法

### 准备工作

确保系统中已安装以下依赖：

1. Python 3.6+
2. matplotlib
3. numpy
4. pandas

### 快速开始

使用Shell脚本运行测试：

```bash
# 赋予脚本执行权限
chmod +x run_comparison.sh

# 运行默认测试（每个框架训练1个epoch）
./run_comparison.sh
```

### 命令行选项

```
使用方法: ./run_comparison.sh [选项]

选项:
  -e, --epochs NUM     设置最大训练轮数（默认：1）
  -i, --iters NUM      设置最大训练迭代数（而不是轮数）
  -o, --output DIR     设置结果输出目录（默认：comparison_results）
  -j, --jittor-only    仅运行Jittor框架
  -p, --pytorch-only   仅运行PyTorch框架
  --jittor-log FILE    指定现有的Jittor训练日志文件（不执行训练）
  --pytorch-log FILE   指定现有的PyTorch训练日志文件（不执行训练）
  -h, --help           显示帮助信息
```

### 示例

1. 仅运行训练10次迭代：

```bash
./run_comparison.sh --iters 10
```

2. 仅测试Jittor框架：

```bash
./run_comparison.sh --jittor-only
```

3. 使用已有的训练日志文件：

```bash
./run_comparison.sh --jittor-log /path/to/jittor/log.txt --pytorch-log /path/to/pytorch/log.txt
```

4. 指定输出目录：

```bash
./run_comparison.sh --output my_test_results
```

## 输出结果

测试完成后，将在指定的输出目录（默认为`comparison_results`）生成以下文件：

1. `framework_comparison_report.md`: 详细的对比报告
2. `loss_comparison.png`: 损失曲线对比图
3. `time_comparison.png`: 训练时间对比图

## 对比内容

测试对比的主要内容包括：

1. **损失函数对比**：
   - 总损失
   - 分类损失（loss_cls）
   - 边界框回归损失（loss_bbox）
   - 分布焦点损失（loss_dfl）

2. **性能对比**：
   - 每次迭代的训练时间
   - 总训练时间

3. **结果对比**：
   - 平均精度（mAP）等评估指标

## 注意事项

1. 为了获得完整的对比结果，建议运行完整的训练过程
2. 对于大型数据集，可以使用`--iters`参数限制训练迭代次数，以快速获得对比结果
3. 如果已有训练日志，可以直接使用`--jittor-log`和`--pytorch-log`参数进行分析 