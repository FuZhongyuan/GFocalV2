# GFL框架对比测试工具

这个工具用于自动化对比PyTorch和Jittor框架下的GFL（Generalized Focal Loss）模型实现的训练效果和结果。

## 功能介绍

该工具可以：

1. 分别在PyTorch和Jittor框架下运行GFL模型的训练
2. 收集训练过程中的日志数据
3. 分析并对比两个框架下的损失、学习率、训练时间和评估结果
4. 生成详细的对比报告和可视化图表

## 文件说明

- `compare_gfl_frameworks.py`: 主要的Python脚本，用于运行对比测试
- `run_comparison.sh`: 简化运行对比测试的Shell脚本
- `README_CN.md`: 本说明文档（中文版）

## 使用方法

### 准备工作

确保系统中已安装以下依赖：

1. Python 3.6+
2. matplotlib
3. numpy
4. pandas

### 运行对比测试

使用Shell脚本运行对比测试：

```bash
# 赋予脚本执行权限
chmod +x run_comparison.sh

# 运行默认测试（每个框架训练1个epoch）
./run_comparison.sh
```

#### 命令行选项

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

#### 1. 运行指定轮数的训练

```bash
./run_comparison.sh --epochs 5
```

这将在两个框架下分别运行5个epoch的训练，并生成对比报告。

#### 2. 指定迭代次数而不是轮数

```bash
./run_comparison.sh --iters 100
```

这将限制训练最多进行100次迭代。

#### 3. 仅运行一个框架

```bash
./run_comparison.sh --jittor-only
```

或

```bash
./run_comparison.sh --pytorch-only
```

#### 4. 使用现有的日志文件

如果您已经有训练日志，可以直接使用它们而不进行训练：

```bash
./run_comparison.sh --jittor-log path/to/jittor/log.txt --pytorch-log path/to/pytorch/log.txt
```

#### 5. 指定输出目录

```bash
./run_comparison.sh --output my_comparison_results
```

## 输出结果

测试完成后，将在指定的输出目录（默认为`comparison_results`）生成以下文件：

1. `framework_comparison_report.md`: 详细的对比报告，包含损失函数、训练性能和评估结果的对比
2. `loss_comparison.png`: 损失曲线对比图
3. `time_comparison.png`: 训练时间对比图
4. `map_comparison.png`: 评估指标(mAP)对比图
5. `map_radar_chart.png`: 评估指标雷达图（当有足够的数据点时）

### 报告内容

对比报告主要包括以下几个部分：

1. **训练记录统计**: 记录两个框架的迭代次数和epoch数
2. **损失函数对比**: 比较两个框架下的总损失和各个损失组件
3. **训练性能对比**: 比较两个框架的训练速度
4. **评估结果对比**: 比较两个框架的mAP和各个子类别的mAP
5. **结论**: 对损失函数对齐情况、训练性能和评估结果进行总结分析

## 注意事项

1. 如果发现日志中有"the testing results of the whole dataset is empty"的错误，需要重新运行测试
2. 如果日志中缺少loss记录，需要修改训练代码添加loss的打印逻辑
3. 为了获得更准确的对比结果，建议使用相同的训练配置和数据集

## 常见问题

1. **Q: 为什么两个框架的损失值有差异？**  
   A: 差异可能来源于初始化方式、计算精度或实现细节的不同。如果差异小于5%，通常认为是正常的浮点误差。

2. **Q: 为什么mAP值为0？**  
   A: 可能是因为训练轮数不足，模型还没有学到有效特征，或者测试过程中出现了问题。

3. **Q: 如何增加报告的详细程度？**  
   A: 可以修改训练代码增加更详细的日志输出，或者增加训练轮数以获得更多的数据点。 