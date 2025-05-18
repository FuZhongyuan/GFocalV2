import jittor as jt
import math
from nets import resnet
from jittor import nn
from nets.common import FPN, CGR, CBR
from losses.gfocal import Project, GFocalLoss


def non_max_suppression(prediction,
                        conf_thresh=0.05,
                        iou_thresh=0.5,
                        max_det=300,
                        max_box=2048,
                        max_layer_num=1000
                        ):
    """
    :param max_layer_num:
    :param prediction:
    :param conf_thresh:
    :param iou_thresh:
    :param max_det:
    :param max_box:
    :return: (x1,y1,x2,y2,score,cls_id)
    """
    bs = prediction[0].shape[0]
    out = [None] * bs
    for bi in range(bs):
        batch_predicts_list = [jt.zeros(size=(0, 6), dtype=prediction[0].dtype).float() for _ in range(len(prediction))]
        for lj in range(len(prediction)):
            one_layer_bath_predict = prediction[lj][bi]
            reg_predicts = one_layer_bath_predict[:, :4]
            cls_predicts = one_layer_bath_predict[:, 4:]

            max_val, max_idx = cls_predicts.max(dim=1)
            valid_bool_idx = max_val > conf_thresh
            if valid_bool_idx.sum() == 0:
                continue
            valid_val = max_val[valid_bool_idx]
            sorted_idx = valid_val.argsort(descending=True)
            valid_val = valid_val[sorted_idx]
            valid_box = reg_predicts[valid_bool_idx, :][sorted_idx]
            valid_cls = max_idx[valid_bool_idx][sorted_idx]
            if 0 < max_layer_num < valid_box.shape[0]:
                valid_val = valid_val[:max_layer_num]
                valid_box = valid_box[:max_layer_num, :]
                valid_cls = valid_cls[:max_layer_num]
            batch_predicts = jt.concat([valid_box, valid_val.unsqueeze(-1), valid_cls.unsqueeze(-1)], dim=-1)
            batch_predicts_list[lj] = batch_predicts
        x = jt.concat(batch_predicts_list, dim=0)
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * max_box
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        # Jittor没有直接对应的nms实现，需要自定义或使用内置函数
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
        
        if i.shape[0] > max_det:
            i = i[:max_det]
        out[bi] = x[i]
    return out


class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(jt.array([init_val]))

    def execute(self, x):
        return x * self.scale


class SequenceCNR(nn.Module):
    def __init__(self,
                 in_channel,
                 inner_channel,
                 kernel_size=3,
                 stride=1,
                 num=4,
                 padding=None,
                 bias=True,
                 block_type='CGR'):
        super(SequenceCNR, self).__init__()
        self.bones = list()
        for i in range(num):
            if i == 0:
                block = eval(block_type)(in_channel, inner_channel, kernel_size, stride, padding=padding, bias=bias)
            else:
                block = eval(block_type)(inner_channel, inner_channel, kernel_size, stride, padding=padding, bias=bias)
            self.bones.append(block)
        self.bones = nn.Sequential(*self.bones)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        return self.bones(x)


class GFocalClsHead(nn.Module):
    def __init__(self,
                 in_channel=256,
                 num_anchors=9,
                 num_cls=80):
        super(GFocalClsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_cls = num_cls
        self.cls = nn.Conv2d(in_channel, self.num_anchors * self.num_cls, 3, 1, 1)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def execute(self, x):
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, self.num_cls).view(bs, -1, self.num_cls)
        return x


class GFocalRegHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9, reg_max=16):
        super(GFocalRegHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_output = 4 * (reg_max + 1)
        self.reg_max = reg_max
        self.reg = nn.Conv2d(in_channel, self.num_anchors * self.num_output, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.reg(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, self.num_output) \
            .view(x.size(0), -1, self.num_output)
        return x


class SubNetFC(nn.Module):
    def __init__(self, m_top_k=4, inner_channel=64, add_mean=True):
        super(SubNetFC, self).__init__()
        self.m_top_k = m_top_k
        self.add_mean = add_mean
        total_dim = (m_top_k + 1) * 4 if add_mean else m_top_k * 4
        self.reg_conf = nn.Sequential(
            nn.Linear(total_dim, inner_channel),
            nn.ReLU(),
            nn.Linear(inner_channel, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        '''
        :param x: [bs,all,4*(reg_max+1)]
        :return:
        quality_score: shape=[bs,all,1]
        '''
        bs, n, c = x.shape
        x = x.view(bs, n, 4, -1)
        origin_type = x.dtype
        if x.dtype == jt.float16:
            x = x.float()
        prob_topk, _ = x.softmax(dim=-1).topk(self.m_top_k, dim=-1)  # shape=[bs,n,4,topk]
        if self.add_mean:
            stat = jt.concat([prob_topk, prob_topk.mean(dim=-1, keepdims=True)], dim=-1)
        else:
            stat = prob_topk
        if stat.dtype != origin_type:
            stat = stat.to(origin_type)
        quality_score = self.reg_conf(stat.reshape(bs, n, -1))
        return quality_score


class GFocalHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel,
                 anchor_sizes,
                 anchor_scales,
                 anchor_ratios,
                 strides,
                 subnet_dim=64,
                 m_top_k=4,
                 add_mean=True,
                 num_cls=80,
                 num_convs=4,
                 layer_num=5,
                 reg_max=16,
                 block_type='CGR'):
        super(GFocalHead, self).__init__()
        self.num_cls = num_cls
        self.layer_num = layer_num
        self.anchor_sizes = anchor_sizes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.strides = strides
        self.reg_max = reg_max
        self.cls_subnet = nn.ModuleList()
        self.reg_subnet = nn.ModuleList()
        self.cls_pred = nn.ModuleList()
        self.reg_pred = nn.ModuleList()
        self.iou_pred = nn.ModuleList()
        self.scales = nn.ModuleList()
        self.gfl_project = Project(reg_max=reg_max)
        total_anchor_num = len(anchor_scales) * len(anchor_ratios)
        for _ in range(layer_num):
            self.cls_subnet.append(SequenceCNR(in_channel, inner_channel, num=num_convs, block_type=block_type))
            self.reg_subnet.append(SequenceCNR(in_channel, inner_channel, num=num_convs, block_type=block_type))

            self.cls_pred.append(GFocalClsHead(in_channel=inner_channel, num_anchors=total_anchor_num, num_cls=num_cls))
            self.reg_pred.append(GFocalRegHead(in_channel=inner_channel, num_anchors=total_anchor_num, reg_max=reg_max))
            self.iou_pred.append(SubNetFC(m_top_k=m_top_k, inner_channel=subnet_dim, add_mean=add_mean))

            self.scales.append(Scale(1.0))

    def build_anchors_delta(self, size=32.):
        anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                w = size * scale * math.sqrt(ratio)
                h = size * scale / math.sqrt(ratio)
                anchors.extend([0, 0, w, h])
        return jt.array(anchors).reshape(-1, 4)

    def build_anchors(self, feature_maps):
        all_anchors = []
        num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        for i, stride in enumerate(self.strides):
            fm_h, fm_w = feature_maps[i].shape[-2:]
            cell_x = jt.arange(0, fm_w * stride, stride) + 0.5 * stride
            cell_y = jt.arange(0, fm_h * stride, stride) + 0.5 * stride
            cell_y, cell_x = jt.meshgrid((cell_y, cell_x))
            cell_xy = jt.stack((cell_x, cell_y), dim=-1).reshape(-1, 2)
            anchors_delta = self.build_anchors_delta(self.anchor_sizes[i])
            layer_anchors = anchors_delta.reshape(1, -1, 4).repeat(fm_h * fm_w, 1, 1)
            layer_anchors[:, :, :2] += cell_xy.unsqueeze(1)
            layer_anchors = layer_anchors.reshape(-1, 4)
            all_anchors.append(layer_anchors)
        return all_anchors, [a.shape[0] for a in all_anchors]

    def execute(self, xs):
        cls_pred_list = []
        reg_pred_list = []
        iou_pred_list = []
        for l_id, (cls_subnet, reg_subnet, cls_pred, reg_pred, iou_pred, scale, feature) in \
                enumerate(zip(self.cls_subnet, self.reg_subnet, self.cls_pred, self.reg_pred, self.iou_pred, self.scales, xs)):
            cls_feature = cls_subnet(feature)
            reg_feature = reg_subnet(feature)
            reg_output = reg_pred(reg_feature)
            cls_output = cls_pred(cls_feature)
            iou_output = iou_pred(reg_output)

            # Jittor处理
            reg_output = scale(reg_output)

            cls_pred_list.append(cls_output.sigmoid() * iou_output.sigmoid())
            reg_pred_list.append(reg_output)
            iou_pred_list.append(iou_output)

        all_anchors, nums_anchor_per_layer = self.build_anchors(xs)
        all_anchors = jt.concat(all_anchors, dim=0)

        # 阶段1 训练阶段
        if self.is_training():
            return cls_pred_list, reg_pred_list, iou_pred_list, all_anchors, nums_anchor_per_layer

        # 阶段2 预测阶段
        clone_cls_pred_list = [out.detach() for out in cls_pred_list]
        clone_reg_pred_list = [out.detach() for out in reg_pred_list]
        
        bs = cls_pred_list[0].shape[0]
        
        decode_regression = self.gfl_project(jt.concat(clone_reg_pred_list, dim=1))
        anchor_points = ((all_anchors[:, :2] + all_anchors[:, 2:]) / 2).unsqueeze(0)
        pred_bboxes = []
        for stride, feature in zip(self.strides, xs):
            fm_h, fm_w = feature.shape[-2:]
            expand_stride = jt.full((fm_h * fm_w * len(self.anchor_scales) * len(self.anchor_ratios), 1), stride)
            pred_bboxes.append(expand_stride)
        pred_bboxes = jt.concat(pred_bboxes, dim=0).unsqueeze(0)

        # 预测坐标
        pred_bboxes = distance2box(anchor_points, decode_regression * pred_bboxes)
        pred_cls = jt.concat(clone_cls_pred_list, dim=1)

        # 预测结果
        input_height, input_width = feature_maps[0].shape[2] * self.strides[0], feature_maps[0].shape[3] * self.strides[0]

        # 转为 xyxy、归一化、限制边界
        detections = jt.concat([pred_bboxes, pred_cls], dim=-1)
        
        # 切分为每层的检测结果
        pred_layers = list()
        for num in nums_anchor_per_layer:
            pred_layers.append(detections[:, :num, :])
            detections = detections[:, num:, :]
        
        return pred_layers


class GFocal(nn.Module):
    def __init__(self, **kwargs):
        super(GFocal, self).__init__()
        if 'anchor_scales' not in kwargs:
            kwargs['anchor_scales'] = [1., 1.25, 1.59]
        if 'anchor_ratios' not in kwargs:
            kwargs['anchor_ratios'] = [1., 0.63, 1.59]
        if 'reg_max' not in kwargs:
            kwargs['reg_max'] = 16
        assert 'backbone' in kwargs
        self.backbone_name = kwargs['backbone']
        self.backbone = eval(f"resnet.{self.backbone_name}")(pretrained=kwargs['pretrained'])
        self.fpn = FPN(*self.backbone.inner_channels, out_channel=256)
        in_channel = 256
        inner_channel = 256
        self.anchor_sizes = kwargs['anchor_sizes']
        self.strides = kwargs['strides']
        self.num_cls = kwargs['num_cls']
        self.top_k = kwargs['top_k']
        self.reg_max = kwargs['reg_max']
        self.beta = kwargs.get('beta', 2.)
        self.iou_type = kwargs.get('iou_type', 'giou')
        self.iou_loss_weight = kwargs.get('iou_loss_weight', 2.0)
        self.reg_loss_weight = kwargs.get('reg_loss_weight', 0.25)
        self.conf_thresh = kwargs.get('conf_thresh', 0.05)
        self.nms_iou_thresh = kwargs.get('nms_iou_thresh', 0.6)
        self.max_det = kwargs.get('max_det', 300)
        self.head = GFocalHead(in_channel, inner_channel, self.anchor_sizes, **kwargs)
        self.gfl_loss = GFocalLoss(top_k=self.top_k,
                                 anchor_num_per_loc=len(kwargs['anchor_scales']) * len(kwargs['anchor_ratios']),
                                 strides=self.strides,
                                 beta=self.beta,
                                 iou_type=self.iou_type,
                                 iou_loss_weight=self.iou_loss_weight,
                                 reg_loss_weight=self.reg_loss_weight,
                                 reg_max=self.reg_max)
        
        if hasattr(self.backbone, 'pretrained') and self.backbone.pretrained:
            self.backbone.load(self.backbone.pretrained)

    def execute(self, x, targets=None):
        features = self.backbone(x)
        features = self.fpn(features)
        out = self.head(features)
        
        if self.is_training():
            cls_pred_list, reg_pred_list, iou_pred_list, all_anchors, nums_anchor_per_layer = out
            
            cls_preds = jt.concat(cls_pred_list, dim=1)
            reg_preds = jt.concat(reg_pred_list, dim=1)
            iou_preds = jt.concat(iou_pred_list, dim=1)
            
            if targets is None:
                # 如果在训练时没有提供target，则只返回预测结果
                return {
                    "cls_preds": cls_preds,
                    "reg_preds": reg_preds,
                    "iou_preds": iou_preds
                }
            
            target = targets['target']
            batch_size = target.shape[0]
            gt_boxes = list()
            for i in range(batch_size):
                gt_boxes.append(target[i][:targets['batch_len'][i]])
                
            losses = self.gfl_loss(cls_preds, reg_preds, iou_preds, all_anchors, gt_boxes, nums_anchor_per_layer)
            
            return losses
        else:
            # 测试时
            return {"predicts": non_max_suppression(out, 
                                                   conf_thresh=self.conf_thresh,
                                                   iou_thresh=self.nms_iou_thresh,
                                                   max_det=self.max_det)}


def distance2box(points, distance):
    x1 = points[:, :, 0] - distance[:, :, 0]
    y1 = points[:, :, 1] - distance[:, :, 1]
    x2 = points[:, :, 0] + distance[:, :, 2]
    y2 = points[:, :, 1] + distance[:, :, 3]
    return jt.stack([x1, y1, x2, y2], -1)