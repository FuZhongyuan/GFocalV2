# GFocalV2 (Jittor版本)
迁移自 [lianghemings' repo](https://github.com/liangheming/gfocal) 的PyTorch实现，用于学习目的。

这是GFocalV2目标检测的非官方Jittor实现，基于论文 [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://arxiv.org/pdf/2011.12885.pdf) by Li, Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang

## 环境要求
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
jittor
```

## 训练
目前仅支持COCO检测数据。

### COCO
* 修改main.py（修改配置文件路径）
```python
from solver.ddp_mix_solver import DDPMixSolver
if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="your own config path") 
    processor.run()
```

* 在*config.yaml*中自定义一些参数
```yaml
model_name: gfocal
data:
  train_annotation_path: data/annotations/instances_train2017.json
#  train_annotation_path: data/annotations/instances_val2017.json
  val_annotation_path: data/annotations/instances_val2017.json
  train_img_root: data/train2017
#  train_img_root: data/val2017
  val_img_root: data/val2017
  max_thresh: 640
  use_crowd: False
  batch_size: 8
  num_workers: 4
  debug: False
  remove_blank: Ture

model:
  num_cls: 80
  anchor_sizes: [32, 64, 128, 256, 512]
  strides: [8, 16, 32, 64, 128]
  backbone: resnet50
  pretrained: jittorhub://resnet50.pkl
  top_k: 9
  reg_max: 16
  beta: 2.0
  iou_type: giou
  iou_loss_weight: 2.0
  reg_loss_weight: 0.25
  conf_thresh: 0.05
  nms_iou_thresh: 0.6
  max_det: 300

optim:
  optimizer: Adam
  lr: 0.0001
  milestones: [24,]
  warm_up_epoch: 0
  weight_decay: 0.0001
  epochs: 24
  sync_bn: True
  amp: True
val:
  interval: 1
  weight_path: weights


gpus: 0,1,2,3
```

* 运行训练脚本
```shell script
python main.py
```

## 参考
官方实现 [GFocalV2](https://github.com/implus/GFocalV2)
```text
@article{li2020generalizedv2,
    title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
    author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
    journal={arXiv preprint},
    year={2020}
}
``` 