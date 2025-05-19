import jittor as jt
import jittor.nn as nn
import numpy as np
import math
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Union

# 适配器函数和类，用于实现PyTorch中的功能

def load_state_dict_from_url(url, progress=True):
    """
    替代PyTorch的load_state_dict_from_url函数，使用Jittor的加载机制
    对于resnet50，我们使用jittorhub上的预训练模型
    """
    print(f"[INFO] Simulating loading pretrained weights from {url}")
    # 这里不实际从URL加载，而是返回空字典，模型初始化时会使用jittorhub
    return {}

class DistributedDataParallel(nn.Module):
    """
    模拟PyTorch的DistributedDataParallel模块
    Jittor自动支持分布式训练，不需要额外的包装器
    """
    def __init__(self, module, device_ids=None, output_device=None):
        super(DistributedDataParallel, self).__init__()
        self.module = module

    def execute(self, *args, **kwargs):
        return self.module(*args, **kwargs)

# 判断模型是否为并行模型
def is_parallel(model):
    return isinstance(model, DistributedDataParallel)

# 复制属性的辅助函数
def copy_attr(a, b, include=(), exclude=()):
    """
    复制模型b的属性到模型a
    Args:
        a: 目标模型
        b: 源模型
        include: 需要包含的属性
        exclude: 需要排除的属性
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
    
class ModelEMA:
    """ 
    模型指数移动平均，参考自https://github.com/rwightman/pytorch-image-models
    保持模型state_dict中所有内容的移动平均值（参数和缓冲区）。
    这旨在提供类似于TensorFlow的ExponentialMovingAverage功能
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    对于某些训练方案，权重的平滑版本是必要的。
    这个类对于在模型初始化、GPU分配和分布式训练包装器序列中的初始化位置很敏感。
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # 创建EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model)
        self.ema.eval()  # FP32 EMA
        self.updates = updates  # EMA更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 指数衰减斜坡（帮助早期epoch）
        # 禁用所有参数的梯度计算
        for p in self.ema.parameters():
            p.stop_grad = True
        
        self.shadow = {}  # 用于兼容性
        self.backup = {}  # 用于兼容性

    def update(self, model):
        # 更新EMA参数
        with jt.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # 模型state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype in [jt.float, jt.float32, jt.float64]:  # 检查是否为浮点类型
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # 更新EMA属性
        copy_attr(self.ema, model, include, exclude)
    
def reduce_sum(tensor):
    """
    模拟PyTorch的reduce_sum函数
    在单机环境下直接返回输入值
    """
    return tensor

class GradScaler:
    """
    模拟PyTorch的GradScaler
    Jittor不需要显式的混合精度训练管理
    """
    def __init__(self, device_type='cuda', enabled=True):
        self.enabled = enabled

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

class autocast:
    """
    模拟PyTorch的autocast上下文管理器
    Jittor自动处理混合精度训练
    """
    def __init__(self, device_type='cuda', enabled=True):
        self.enabled = enabled
        self.device_type = device_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# nms函数实现
def nms(boxes, scores, iou_threshold):
    """
    实现非极大值抑制
    Args:
        boxes (Var): 边界框 [N, 4]
        scores (Var): 得分 [N]
        iou_threshold (float): IoU阈值
    Returns:
        keep (Var): 保留的边界框索引
    """
    # 确保输入是jittor变量
    boxes = jt.array(boxes) if not isinstance(boxes, jt.Var) else boxes
    scores = jt.array(scores) if not isinstance(scores, jt.Var) else scores
    
    # 获取排序后的得分索引
    _, order = jt.argsort(scores, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)
            
        # 计算IoU
        xx1 = jt.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = jt.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = jt.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = jt.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = jt.maximum(0.0, xx2 - xx1)
        h = jt.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        
        iou = inter / (area1 + area2 - inter)
        
        inds = jt.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # +1 因为inds是从order[1:]计算的
    
    return jt.array(keep)

# Pytorch分布式训练的模拟
class dist:
    @staticmethod
    def is_initialized():
        return False
    
    @staticmethod
    def get_rank():
        return 0
    
    @staticmethod
    def init_process_group(backend=None, init_method=None):
        print("[INFO] Jittor不需要显式初始化分布式进程组")
    
    @staticmethod
    def destroy_process_group():
        pass

# 数据加载器适配
class DistributedSampler:
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        
    def __iter__(self):
        return iter(range(len(self.dataset)))
        
    def __len__(self):
        return len(self.dataset)

# 将tensor移动到设备的适配函数
def to_device(tensor, device):
    """
    模拟PyTorch的.to(device)方法
    Jittor不需要显式的设备管理
    """
    return tensor 