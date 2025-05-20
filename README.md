# GFL框架对比测试工具

这个工具用于自动化对比PyTorch和Jittor框架下的GFL（Generalized Focal Loss）模型实现的训练效果、推理性能和结果。

## 功能介绍

该工具可以：

1. 分别在PyTorch和Jittor框架下运行GFL模型的训练和推理
2. 收集训练和推理过程中的日志数据
3. 分析并对比两个框架下的训练损失、推理性能和结果
4. 生成对比报告和可视化图表

## 文件说明

- `compare_gfl_frameworks.py`: 训练过程对比的Python脚本
- `run_comparison.sh`: 简化运行训练对比测试的Shell脚本
- `compare_gfl_inference.py`: 推理性能对比的Python脚本
- `run_inference_comparison.sh`: 简化运行推理对比测试的Shell脚本
- `README.md`: 本说明文档

## 使用方法

### 准备工作

确保系统中已安装以下依赖：

1. Python 3.6+
2. matplotlib
3. numpy
4. pandas

### 训练对比测试

使用Shell脚本运行训练对比测试：

```bash
# 赋予脚本执行权限
chmod +x run_comparison.sh

# 运行默认测试（每个框架训练1个epoch）
./run_comparison.sh
```

#### 训练对比命令行选项

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

### 推理对比测试

使用Shell脚本运行推理对比测试：

```bash
# 赋予脚本执行权限
chmod +x run_inference_comparison.sh

# 运行默认测试（使用最新的检查点文件）
./run_inference_comparison.sh
```

#### 推理对比命令行选项

```
使用方法: ./run_inference_comparison.sh [选项]

选项:
  -j, --jittor-checkpoint PATH    指定Jittor框架的检查点文件路径
  -p, --pytorch-checkpoint PATH   指定PyTorch框架的检查点文件路径
  -b, --batch-size NUM           设置推理批次大小（默认：1）
  -n, --num-samples NUM          限制测试样本数量
  -o, --output DIR               设置结果输出目录（默认：inference_comparison_results）
  --jittor-only                  仅测试Jittor框架
  --pytorch-only                 仅测试PyTorch框架
  --jittor-log FILE              指定现有的Jittor推理日志文件（不执行推理）
  --pytorch-log FILE             指定现有的PyTorch推理日志文件（不执行推理）
  -h, --help                     显示帮助信息
```

## 示例

### 训练对比示例

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

### 推理对比示例

1. 指定检查点文件：

```bash
./run_inference_comparison.sh --jittor-checkpoint /path/to/jittor_checkpoint.pkl --pytorch-checkpoint /path/to/pytorch_checkpoint.pth
```

2. 仅测试100个样本：

```bash
./run_inference_comparison.sh --num-samples 100
```

3. 使用批处理进行推理：

```bash
./run_inference_comparison.sh --batch-size 4
```

## 输出结果

### 训练对比结果

测试完成后，将在指定的输出目录（默认为`comparison_results`）生成以下文件：

1. `framework_comparison_report.md`: 详细的训练对比报告
2. `loss_comparison.png`: 损失曲线对比图
3. `time_comparison.png`: 训练时间对比图

### 推理对比结果

测试完成后，将在指定的输出目录（默认为`inference_comparison_results`）生成以下文件：

1. `inference_comparison_report.md`: 详细的推理对比报告
2. `inference_time_comparison.png`: 推理时间分布对比图
3. `inference_time_boxplot.png`: 推理时间箱线图
4. `ap_comparison.png`: 检测精度(mAP)对比图

## 对比内容

### 训练对比

训练对比的主要内容包括：

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

### 推理对比

推理对比的主要内容包括：

1. **性能对比**：
   - 平均推理时间
   - 帧率(FPS)
   - 推理时间分布

2. **精度对比**：
   - mAP
   - mAP@0.5
   - mAP@0.75
   - 不同尺寸物体的mAP

## 注意事项

1. 为了获得完整的对比结果，建议首先运行训练，然后使用训练生成的检查点进行推理测试
2. 对于大型数据集，可以使用`--iters`参数限制训练迭代次数，以快速获得对比结果
3. 如果已有训练日志或推理日志，可以直接使用`--jittor-log`和`--pytorch-log`参数进行分析
4. 推理测试默认会自动查找最新的检查点文件，如果要使用特定的检查点，请使用`--jittor-checkpoint`和`--pytorch-checkpoint`参数指定