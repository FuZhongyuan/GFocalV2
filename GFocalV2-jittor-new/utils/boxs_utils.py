import jittor as jt
import numpy as np
from tqdm import tqdm


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clamp(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clamp(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clamp(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clamp(0, img_shape[0])  # y2


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.transpose())
    area2 = box_area(box2.transpose())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (jt.minimum(box1.unsqueeze(1)[:, :, 2:], box2[:, 2:]) - jt.maximum(box1.unsqueeze(1)[:, :, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1.unsqueeze(1) + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xyxy2xywh(x):
    wh = x[..., [2, 3]] - x[..., [0, 1]]
    xcyc = x[..., [0, 1]] + wh * 0.5
    if isinstance(x, jt.Var):
        ret = jt.concat([xcyc, wh], dim=-1)
    elif isinstance(x, np.ndarray):
        ret = np.concatenate([xcyc, wh], axis=-1)
    else:
        raise NotImplementedError()

    return ret


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x1y1 = x[..., [0, 1]] - x[..., [2, 3]] * 0.5
    x2y2 = x1y1 + x[..., [2, 3]]
    if isinstance(x, jt.Var):
        ret = jt.concat([x1y1, x2y2], dim=-1)
    elif isinstance(x, np.ndarray):
        ret = np.concatenate([x1y1, x2y2], axis=-1)
    else:
        raise NotImplementedError()
    return ret


def non_max_suppression(prediction,
                        conf_thresh=0.01,
                        iou_thresh=0.6,
                        max_det=300):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns: detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        # 应用置信度阈值
        xc = x[:, 4] > conf_thresh
        x = x[xc]
        
        # 如果没有检测
        if not x.shape[0]:
            continue
            
        # 从(center x, center y, width, height)转为(x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        x = jt.concat([box, x[:, [4]]], dim=-1)
        
        # 执行NMS
        boxes, scores = x[:, :4], x[:, 4]
        
        # 实现NMS算法
        keep = []
        order = jt.argsort(scores, descending=True)
        
        while order.shape[0] > 0:
            i = order[0] # 选择得分最高的边界框
            keep.append(i)
            
            # 计算当前边界框与其他边界框的IoU
            if order.shape[0] == 1:
                break
            
            remaining_boxes = boxes[order[1:]]
            current_box = boxes[i].unsqueeze(0)
            ious = box_iou(current_box, remaining_boxes)[0]
            
            # 排除IoU大于阈值的框
            idx = jt.where(ious <= iou_thresh)[0]
            order = order[idx + 1]  # +1因为我们跳过了第一个框
            
        keep = jt.concat(keep) if len(keep) > 1 else keep[0]
        
        # 如果检测结果过多，限制数量
        if keep.shape[0] > max_det:  
            keep = keep[:max_det]
            
        output[xi] = x[keep]
        
    return output 