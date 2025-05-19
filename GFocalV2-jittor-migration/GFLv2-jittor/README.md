# GFocalV2-Jittor

这是一个[Jittor](https://cg.cs.tsinghua.edu.cn/jittor/)实现的GFocalV2目标检测模型，基于[Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://arxiv.org/pdf/2011.12885.pdf)论文，由李祥、王文海、胡晓林、李军、唐金辉和杨健发布。

## 环境要求
```text
jittor
pyyaml
numpy
opencv-python
pycocotools
```

## 安装Jittor
请参考[Jittor官方文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)安装Jittor框架。

## 项目结构
```
jittor/
├── config/               # 配置文件目录
├── datasets/             # 数据集加载模块
├── losses/               # 损失函数实现
├── metrics/              # 评估指标实现
├── nets/                 # 网络模型实现
│   ├── common.py         # 公共网络组件
│   ├── resnet.py         # ResNet骨干网络
│   └── retinanet.py      # GFocal检测器
├── patches/              # PyTorch到Jittor的适配补丁
│   └── torch_adapter.py  # 适配器，实现PyTorch特有API
├── solver/               # 训练求解器
│   └── ddp_mix_solver.py # 分布式训练求解器
├── utils/                # 工具函数
├── main.py               # 主程序入口
└── README.md             # 本文件
```

## 预训练模型
本实现使用Jittor的预训练ResNet-50模型，预训练模型URL: `jittorhub://resnet50.pkl`

## 训练
### COCO数据集
1. 准备COCO数据集
```bash
mkdir -p data/coco
# 下载COCO 2017数据集并解压到data/coco目录
```

2. 修改配置文件 `config/gfocal.yaml`
```yaml
data:
  train_annotation_path: ./data/coco/annotations/instances_train2017.json
  val_annotation_path: ./data/coco/annotations/instances_val2017.json
  train_img_root: ./data/coco/train2017
  val_img_root: ./data/coco/val2017
```

3. 运行训练脚本
```bash
python main.py
```

## 与PyTorch版本的区别
- 使用Jittor框架替代PyTorch
- 预训练模型从`jittorhub://resnet50.pkl`加载
- 不再需要显式的分布式处理
- 混合精度训练由Jittor自动处理
- 设备管理由Jittor自动处理

## 转换说明
本项目是从PyTorch版本的GFocalV2转换而来，主要改动包括：
1. 将PyTorch特有的API替换为Jittor的对应实现
2. 使用Jittor的预训练模型加载机制
3. 适配Jittor的数据加载和处理方式
4. 实现一个torch_adapter模块，模拟PyTorch的一些特有功能

## 参考
原始实现 [GFocalV2](https://github.com/implus/GFocalV2)
```text
@article{li2020generalizedv2,
    title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
    author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
    journal={arXiv preprint},
    year={2020}
}
``` 