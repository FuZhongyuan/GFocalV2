# GFocalV2 Jittor

## 简介

这是一个基于 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 框架实现的 GFocalV2 目标检测模型。GFocalV2（Generalized Focal Loss V2）是一种高效的目标检测方法，通过改进的分类和回归损失函数提高检测精度。

## 支持的模型

- GFocalV2

## 安装

请先按照 [教程](https://github.com/Jittor/jittor) 安装 jittor 框架。
推荐使用已测试过的 jittor==1.3.6.10。

然后，通过运行以下命令安装 `jittordet`：
```
pip install -v -e .
```

如果你想使用多GPU训练或测试，请安装 OpenMPI：
```
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

## 快速开始

我们提供了一个简单的示例脚本 `example.py`，可以轻松地进行 GFocalV2 训练和测试：

```bash
# 训练
python example.py --mode train --config configs/gfl/gfl_r50_fpn_coco_1x.yml

# 测试
python example.py --mode test --config configs/gfl/gfl_r50_fpn_coco_1x.yml
```

## 训练

支持单GPU和多GPU训练：
```
# 单GPU
python tools/train.py {CONFIG_PATH}

# 多GPU
bash tools/dist_train.sh {CONFIG_PATH} {NUM_GPUS}
```

## 测试

支持单GPU和多GPU测试：
```
# 单GPU
python tools/test.py {CONFIG_PATH}

# 多GPU
bash tools/dist_test.sh {CONFIG_PATH} {NUM_GPUS}
```

## 引用

如果您发现这个工作对您的研究有帮助，请考虑引用以下条目：

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

@article{li2020generalizedv2,
    title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
    author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
    journal={arXiv preprint},
    year={2020}
}
```

## 致谢

我们的代码基于以下开源代码库开发：

- [Jittor](https://github.com/Jittor/jittor)
- [JDet](https://github.com/Jittor/JDet)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

我们衷心感谢他们的卓越工作。
