# GFocalV2 框架内存占用分析工具

本工具用于比较 Jittor 和 PyTorch 框架在运行 GFocalV2 模型训练和测试时的内存占用情况。

## 功能特点

- 实时监控 CPU 和 GPU 内存使用情况
- 生成详细的内存使用数据和图表
- 提供框架间内存使用对比报告
- 支持自定义训练和测试参数

## 需求

- Python 3.6+
- psutil
- matplotlib
- pandas
- NVIDIA GPU（可选，用于监控GPU内存）

## 文件说明

- `compare_memory_usage.py`: 主要的Python脚本，负责监控和比较内存使用
- `run_memory_comparison.sh`: Shell脚本，用于安装依赖并运行比较脚本
- `MEMORY_ANALYSIS_README.md`: 本文档

## 使用方法

### 快速开始

1. 确保已安装所需的依赖包（脚本会自动安装）
2. 运行Shell脚本：

```bash
./run_memory_comparison.sh
```

### 自定义配置

如需调整测试参数（如训练迭代次数、测试样本数量），请修改 `compare_memory_usage.py` 中的以下内容：

```python
# 运行测试命令（为了演示，我们只运行几秒钟）
jittor_train_cmd += " --max-iters 20"  # 修改为所需的迭代次数
pytorch_train_cmd += " --max-iters 20"  # 修改为所需的迭代次数

jittor_test_cmd += " --eval-options 'max_det=100'"  # 修改为所需的检测数量
pytorch_test_cmd += " --eval-options 'max_det=100'"  # 修改为所需的检测数量
```

## 输出结果

执行脚本后，将在 `memory_comparison_results` 目录中生成以下文件：

1. **内存使用摘要CSV文件**：`memory_comparison_summary.csv`
   - 包含各框架在训练和测试模式下的最大/平均内存使用值

2. **内存使用对比图表**：`memory_comparison_summary.png`
   - 显示各框架在不同模式下的内存使用对比

3. **详细内存数据文件**：`[框架]_[模式]_[时间戳].json`
   - 包含每次采样的详细内存数据

4. **内存使用曲线图**：`[框架]_[模式]_[时间戳].png`
   - 显示内存使用随时间变化的曲线

5. **运行日志**：`[框架]_[模式]_log.txt`
   - 包含训练/测试过程的输出日志

## 数据解释

### 内存指标说明

- **max_ram_mb**: 最大RAM内存使用量（MB）
- **avg_ram_mb**: 平均RAM内存使用量（MB）
- **max_gpu_mb**: 最大GPU内存使用量（MB）
- **avg_gpu_mb**: 平均GPU内存使用量（MB）
- **duration_seconds**: 执行持续时间（秒）

### 图表说明

生成的图表包含四个子图，分别显示：
1. RAM最大使用量比较
2. RAM平均使用量比较
3. GPU最大使用量比较
4. GPU平均使用量比较

每个子图都包含训练和测试两种模式下的数据对比。

## 常见问题

1. **Q: 如何监控完整的训练过程？**
   A: 修改`compare_memory_usage.py`中的`--max-iters`参数或完全删除该参数。

2. **Q: GPU内存数据为0怎么办？**
   A: 确保系统中安装了NVIDIA GPU和驱动，且`nvidia-smi`命令可用。

3. **Q: 可以只监控特定阶段（如只监控训练）吗？**
   A: 可以，修改`compare_memory_usage.py`中的`main()`函数，注释掉不需要监控的部分。

## 注意事项

- 监控过程会增加少量额外的系统开销
- 默认采样间隔为0.5秒，可根据需要在代码中调整
- 如需监控长时间运行的任务，请确保系统有足够的磁盘空间存储数据 