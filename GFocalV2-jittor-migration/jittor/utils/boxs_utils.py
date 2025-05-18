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
    inter = (jt.minimum(box1[:, None, 2:], box2[:, 2:]) - jt.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


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
    if prediction.dtype == jt.float16:
        prediction = prediction.float()
    xc = prediction[..., 4] > conf_thresh
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box = xywh2xyxy(x[:, :4])
        x = jt.concat([box, x[:, [4]]], dim=-1)
        boxes, scores = x[:, :4], x[:, 4]
        
        # 自定义NMS实现
        keep = []
        order = scores.argsort(descending=True)
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            
            xx1 = jt.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = jt.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = jt.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = jt.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = jt.maximum(0.0, xx2 - xx1)
            h = jt.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算IoU
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area1 + area2 - inter
            iou = inter / union
            
            inds = jt.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        
        i = jt.array(keep)
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output 