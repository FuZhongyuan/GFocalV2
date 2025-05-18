import jittor as jt
import numpy as np
from matplotlib import pyplot as plt
import torch  # 用于Numpy和Tensor之间的转换


def compute_ap(recall, precision):
    """
    计算每个类别的平均精度值
    Args:
        recall: 召回率
        precision: 精确率

    Returns:
        每个类别的平均精度值
    """
    # 通过在0和1之间添加起始值来确保PR曲线包含(0,0)和(1,0)
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # 计算precision包络线
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算召回率变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算各个区域面积的和
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    计算每个类别的AP，需要按照conf进行排序
    Args:
        tp: (N,1) 列表，标记每个预测是否为TP
        conf: (N,) 列表，每个预测的置信度
        pred_cls: (N,) 列表，每个预测的类别
        target_cls: (N,) 列表，每个真实标签的类别

    Returns:
        p,r,ap,f1,unique_classes
    """
    # 按照conf降序排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到每个类别的唯一类别
    unique_classes = np.unique(target_cls)

    # 创建准确率, 召回率, F1, AP 数组
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # 0.1 IoU的PR曲线
    s = [unique_classes.shape[0], tp.shape[1]]  # 每个类别中每个IoU阈值的个数
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)

    # 计算每个类别的PR曲线
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # 真实标签中类别c的个数
        n_p = i.sum()  # 预测中类别c的个数

        if n_p == 0 or n_l == 0:
            continue
        else:
            # 累计tp
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # 召回率
            recall = tpc / (n_l + 1e-16)  # 召回率
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # 精确率
            precision = tpc / (tpc + fpc)  # 精确率
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # 绘制
            py.append(np.interp(px, recall[:, 0], precision[:, 0]))  # 精确率-召回率曲线

    # 计算F1 score
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def coco_map(predicts, targets, imgIds, catIds, maxDets=None):
    """
    计算多个类别的mAP
    Args:
        predicts: [List] 预测的标注，包含[boxes,scores,classes]
        targets: [List] 真实的标注数据集，包含[boxes,classes]
        imgIds: [List] 图片的id
        catIds: [List] 类别的id
        maxDets: [List] 一张图片中预测的最大个数

    Returns:
        dictionary包含如下字段:
        1. IoU_lo_thresh: 计算TP时使用的最小IoU阈值
        2. IoU_hi_thresh: 计算TP时使用的最大IoU阈值
        3. IoUs: 实际TP时使用的IoU阈值
        4. aps: 每个类别不同IoU下的平均精度，形状为(nc, 10), 每个类别对应10个不同IoU下的AP
        5. mAP: 所有类别的平均精度，也就是AP, 形状为(10,)
        6. precision: 各个类别在不同IoU下的P值，形状为(nc, 10)
        7. recall: 各个类别在不同IoU下的R值，形状为(nc, 10)
    """
    p = []
    r = []
    f = []
    a = []
    IoUs = np.arange(0.5, 1, 0.05)
    # IoUs = np.array([0.5])
    if maxDets is None:
        maxDets = [1, 10, 100]
    # 统计每个类别的真实框的个数
    n_class_target = np.zeros(len(catIds))
    for target in targets:
        if len(target["labels"]) != 0:
            for cat_id in target["labels"]:
                index = catIds.index(cat_id)
                n_class_target[index] += 1

    for IoU in IoUs:
        tp = []
        score = []
        pred_c = []
        target_c = []
        for predict_anno, target_anno in zip(predicts, targets):
            predict_box = predict_anno["boxes"]
            predict_score = predict_anno["scores"]
            predict_cls = predict_anno["classes"]
            target_box = target_anno["boxes"]
            target_cls = target_anno["labels"]
            if len(predict_box) == 0:
                continue
            if len(predict_box) > maxDets[-1]:
                sorted_ind = np.argsort(-predict_score)
                predict_score = predict_score[sorted_ind][:maxDets[-1]]
                predict_box = predict_box[sorted_ind][:maxDets[-1]]
                predict_cls = predict_cls[sorted_ind][:maxDets[-1]]
            if len(target_box) != 0:
                for pc, pbox in zip(predict_cls, predict_box):
                    max_target_box = []
                    max_target_cls = []
                    for tc, tbox in zip(target_cls, target_box):
                        if tc == pc:
                            max_target_box.append(tbox)
                            max_target_cls.append(tc)
                    if len(max_target_box) != 0:
                        max_target_box = np.stack(max_target_box)
                        max_IoU = box_iou(pbox, max_target_box)
                        max_index = np.argmax(max_IoU, axis=1)
                        if max_IoU[max_index] >= IoU:
                            if tp and tp[-1].size > 0:
                                b = max_target_box[max_index]
                                b = b.reshape(-1)
                                bj = np.zeros(tp[-1].shape)
                                for i in range(len(tp)):
                                    tpc = target_c[i]
                                    if tpc[0] != pc:  # 不同类别不计算重复
                                        continue
                                    bi = max_target_box[max_index]
                                    bi = bi.reshape(-1)
                                    if all(bi == b):  # 如果两个目标框是一样的(重复的)
                                        bj[i] = 1
                                if not bj.any():  # 如果当前目标框不是重复的, 计为TP
                                    tp.append(np.array([1.]))
                                    target_c.append(np.array([pc]))
                                else:  # 如果当前目标框是重复的，计为FP
                                    tp.append(np.array([0.]))
                                    target_c.append(np.array([pc]))
                            else:  # 第一个预测值如果满足IoU直接计为TP
                                tp.append(np.array([1.]))
                                target_c.append(np.array([pc]))
                        else:  # 如果没有满足IoU阈值的，计为FP
                            tp.append(np.array([0.]))
                            target_c.append(np.array([pc]))
                    else:  # 如果预测的类别对应的target不存在，那么计为FP
                        tp.append(np.array([0.]))
                        target_c.append(np.array([pc]))
                    score.append(predict_score[list(predict_cls).index(pc)])
                    pred_c.append(pc)
            else:  # 如果target是空的，那么所有的predict都是FP
                for pc, ps in zip(predict_cls, predict_score):
                    tp.append(np.array([0.]))
                    target_c.append(np.array([pc]))
                    score.append(ps)
                    pred_c.append(pc)
        if tp:
            # 需要统计所有图片的预测结果
            tp = np.array(tp)
            score = np.array(score)
            pred_c = np.array(pred_c)
            target_c = np.concatenate(target_c, axis=0)
            try:
                p_c, r_c, ap_c, f1, _ = ap_per_class(tp, score, pred_c, target_c)
                # 只统计数据集中的类别的AP
                p.append(p_c[:len(catIds)])
                r.append(r_c[:len(catIds)])
                a.append(ap_c[:len(catIds)])
                f.append(f1[:len(catIds)])
            except:
                p.append(np.zeros(len(catIds)))
                r.append(np.zeros(len(catIds)))
                a.append(np.zeros(len(catIds)))
                f.append(np.zeros(len(catIds)))
    # 对mAP做平均
    p, r, a, f = np.array(p), np.array(r), np.array(a), np.array(f)
    p = p.reshape(len(IoUs), -1).transpose()
    r = r.reshape(len(IoUs), -1).transpose()
    a = a.reshape(len(IoUs), -1).transpose()
    f = f.reshape(len(IoUs), -1).transpose()
    return {
        "IoU_lo_thresh": 0.5,
        "IoU_hi_thresh": 0.95,
        "IoUs": IoUs,
        "precision": p[:, 0:len(IoUs)],
        "recall": r[:, 0:len(IoUs)],
        "AP": a[:, 0:len(IoUs)],
        "F1": f[:, 0:len(IoUs)],
        "mAP": np.mean(a, axis=0).tolist(),
        "n_class": n_class_target.tolist()
    }


def box_iou(box1, box2):
    """
    Return intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    Args:
        box1 (tensor): Boxes, shape: [N, 4]
        box2 (tensor): Boxes, shape: [M, 4]

    Return:
        iou (tensor): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter) 