import math
import random
import cv2 as cv
import numpy as np
import jittor as jt
from copy import deepcopy

cv.setNumThreads(0)


def mosaic_augment(img, target, img_root, annotations, img_idx, imgsz=640, usage_prob=0.5, mixup_prob=0.15, keep_ratio=True):
    # 这是一个简化的Mosaic实现，仅用于占位
    return img, target


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """调整图像大小并添加字母框，保持原始纵横比"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def augment_hsv(img, h_gain=0.014, s_gain=0.68, v_gain=0.36):
    """HSV色彩空间增强"""
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val)))
    img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
    return img


class MOSAIC:
    def __init__(self, img_size, usage_prob, Aug, img_root, imgsz, keep_ratio, annos, mixup_prob):
        self.img_size = img_size
        self.usage_prob = usage_prob
        self.Aug = Aug
        self.img_root = img_root
        self.imgsz = imgsz
        self.keep_ratio = keep_ratio
        self.annos = annos
        self.mixup_prob = mixup_prob

    def __call__(self, img, target, annotations=None, img_idx=None):
        # 简化实现，实际项目中可以根据需要完善
        if random.random() < self.usage_prob:
            return mosaic_augment(img, target, self.img_root, annotations, img_idx, 
                                 self.imgsz, self.usage_prob, self.mixup_prob, self.keep_ratio)
        return img, target


class YOLOV5Augmentation:
    def __init__(self, degrees=0, translate=0, scale=None, shear=0, perspective=0.0,
                 fill_color=114, hsv_h=0.014, hsv_s=0.68, hsv_v=0.36):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale if scale is not None else [0.1, 2.0]
        self.shear = shear
        self.perspective = perspective
        self.fill_color = fill_color
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v

    def __call__(self, img, boxes=None):
        # 实际项目中可以根据需要实现更复杂的增强
        if random.random() < 0.5:
            img = augment_hsv(img, self.hsv_h, self.hsv_s, self.hsv_v)
        return img, boxes


class SSDBaseTransform:
    def __init__(self, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, img, boxes=None):
        # 调整图像大小
        img_resized, ratio, pad = letterbox(img, self.size, auto=False)
        
        # 归一化
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = (img_resized - self.mean) / self.std
        
        if boxes is not None:
            # 调整框的坐标
            boxes = np.array(boxes)
            if boxes.size > 0:
                boxes[:, 0] = boxes[:, 0] * ratio[0] + pad[0]  # x1
                boxes[:, 1] = boxes[:, 1] * ratio[1] + pad[1]  # y1
                boxes[:, 2] = boxes[:, 2] * ratio[0] + pad[0]  # x2
                boxes[:, 3] = boxes[:, 3] * ratio[1] + pad[1]  # y2
                
                # 提取标签
                labels = boxes[:, 4]
                boxes = boxes[:, :4]
                
                return img_resized, boxes, labels
        
        return img_resized, np.array([]), np.array([])


class SSDAugmentationTrain:
    def __init__(self, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
                variance=(0.1, 0.2), use_base=False):
        self.size = size
        self.mean = mean
        self.std = std
        self.variance = variance
        self.use_base = use_base
        self.base_transform = SSDBaseTransform(size, mean, std)

    def __call__(self, img, boxes=None):
        if self.use_base or random.random() > 0.5:
            return self.base_transform(img, boxes)
        
        # 简单的数据增强
        if random.random() < 0.5:
            img = augment_hsv(img, 0.014, 0.68, 0.36)
            
        img_resized, ratio, pad = letterbox(img, self.size, auto=False)
        
        # 归一化
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = (img_resized - self.mean) / self.std
        
        if boxes is not None:
            # 调整框的坐标
            boxes = np.array(boxes)
            if boxes.size > 0:
                boxes[:, 0] = boxes[:, 0] * ratio[0] + pad[0]  # x1
                boxes[:, 1] = boxes[:, 1] * ratio[1] + pad[1]  # y1
                boxes[:, 2] = boxes[:, 2] * ratio[0] + pad[0]  # x2
                boxes[:, 3] = boxes[:, 3] * ratio[1] + pad[1]  # y2
                
                # 提取标签
                labels = boxes[:, 4]
                boxes = boxes[:, :4]
                
                return img_resized, boxes, labels
                
        return img_resized, np.array([]), np.array([]) 