# 深度学习框架性能分析工具

## 简介

这是一个用于比较不同深度学习框架（Jittor和PyTorch）在训练GFocalV2目标检测模型时性能差异的分析工具。工具通过解析训练日志，提取关键性能指标和评估结果，生成可视化图表和详细报告。

## 功能特点

- 自动解析Jittor和PyTorch的训练日志
- 提取和比较关键训练指标（损失函数、学习率等）
- 提取和比较模型评估指标（mAP等）
- 生成多种可视化图表（折线图、柱状图、雷达图等）
- 生成详细的分析报告（Markdown格式）

## 安装依赖

```bash
pip install matplotlib numpy pandas seaborn
```

## 使用方法

1. 运行提供的分析脚本：

```bash
# 方法一：通过shell脚本运行
chmod +x run_analysis.sh
./run_analysis.sh

# 方法二：直接运行Python脚本
python analyze_framework_logs.py --jittor-log "/path/to/jittor/log" --pytorch-log "/path/to/pytorch/log" --output-dir "output_folder"
```

2. 查看生成的报告和可视化结果：

```bash
cd framework_comparison
# 查看README.md和visualization目录下的图表
```

## 输出内容

运行脚本后，将在指定的输出目录中生成以下内容：

- `README.md`：详细的分析报告
- `visualization/`：包含各种可视化图表
  - `loss_comparison.png`：损失函数对比
  - `map_comparison.png`：mAP指标对比
  - `performance_comparison.png`：性能指标对比
  - `speed_comparison.png`：速度对比
  - `radar_comparison.png`：多维度性能雷达图

## 分析维度

工具从以下多个维度对两个框架进行比较：

1. **训练性能**
   - 训练速度（每迭代耗时）
   - 损失函数收敛速度和稳定性
   - 学习率调度

2. **模型精度**
   - 整体mAP
   - mAP@0.5（IoU=0.5时的mAP）
   - mAP@0.75（IoU=0.75时的mAP）
   - 不同尺度物体的检测精度（小、中、大物体）

3. **综合评估**
   - 多维度性能雷达图
   - 性能提升百分比

## 定制化

如需定制分析内容，可以修改以下文件：

- `analyze_framework_logs.py`：调整解析逻辑和可视化样式
- `run_analysis.sh`：修改输入日志路径和输出目录