import jittor as jt
import math
from nets import resnet
import jittor.nn as nn
from nets.common import FPN, CGR, CBR
from utils.boxs_utils import non_max_suppression
from losses.gfocal import Project, GFocalLoss


class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(jt.array(init_val), requires_grad=True)

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
            if isinstance(m, nn.Conv):
                nn.init.gauss_(m.weight, std=0.01)
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
        self.cls = nn.Conv(in_channel, self.num_anchors * self.num_cls, 3, 1, 1)
        nn.init.gauss_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def execute(self, x):
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, h, w, self.num_anchors, self.num_cls).reshape(bs, -1, self.num_cls)
        return x


class GFocalRegHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9, reg_max=16):
        super(GFocalRegHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_output = 4 * (reg_max + 1)
        self.reg_max = reg_max
        self.reg = nn.Conv(in_channel, self.num_anchors * self.num_output, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.gauss_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.reg(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, h, w, self.num_anchors, self.num_output).reshape(x.size(0), -1, self.num_output)
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
                nn.init.gauss_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        '''
        :param x: [bs,all,4*(reg_max+1)]
        :return:
        quality_score: shape=[bs,all,1]
        '''
        bs, n, c = x.shape
        x = x.reshape(bs, n, 4, -1)
        origin_type = x.dtype
        if x.dtype == 'float16':
            x = x.float()
        prob = jt.nn.softmax(x, dim=-1)
        prob_topk, _ = prob.topk(self.m_top_k, dim=-1)  # shape=[bs,n,4,topk]
        if self.add_mean:
            stat = jt.concat([prob_topk, prob_topk.mean(dim=-1, keepdims=True)], dim=-1)
        else:
            stat = prob_topk
        if stat.dtype != origin_type:
            stat = stat.astype(origin_type)
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

        self.anchor_nums = len(self.anchor_scales) * len(self.anchor_ratios)
        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in range(self.layer_num)])
        self.anchors = [jt.zeros((0, 4))] * self.layer_num

        self.cls_bones = SequenceCNR(in_channel, inner_channel,
                                   kernel_size=3, stride=1,
                                   num=num_convs, block_type=block_type)
        self.reg_bones = SequenceCNR(in_channel, inner_channel,
                                   kernel_size=3, stride=1,
                                   num=num_convs, block_type=block_type)
        self.cls_head = GFocalClsHead(inner_channel, self.anchor_nums, num_cls)
        self.reg_head = GFocalRegHead(inner_channel, self.anchor_nums, reg_max=reg_max)
        self.reg_conf = SubNetFC(m_top_k=m_top_k, inner_channel=subnet_dim, add_mean=add_mean)
        self.project = Project(reg_max)

    def build_anchors_delta(self, size=32.):
        """
        :param size:
        :return: [anchor_num, 4]
        """
        scales = jt.array(self.anchor_scales).float()
        ratio = jt.array(self.anchor_ratios).float()
        scale_size = (scales * size)
        w = (scale_size[:, None] * jt.sqrt(ratio[None, :])).reshape(-1) / 2
        h = (scale_size[:, None] / jt.sqrt(ratio[None, :])).reshape(-1) / 2
        delta = jt.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert self.layer_num == len(feature_maps)
        assert len(self.anchor_sizes) == len(feature_maps)
        assert len(self.anchor_sizes) == len(self.strides)
        anchors = list()
        for stride, size, feature_map in zip(self.strides, self.anchor_sizes, feature_maps):
            # 9*4
            anchor_delta = self.build_anchors_delta(size)
            _, _, ny, nx = feature_map.shape
            yv, xv = jt.meshgrid([jt.arange(ny), jt.arange(nx)])
            # h,w,4
            grid = jt.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.reshape(-1, 4)
            anchors.append(anchor)
        return anchors

    def execute(self, xs):
        cls_outputs = list()
        reg_outputs = list()
        for j, x in enumerate(xs):
            cls_tower = self.cls_bones(x)
            reg_tower = self.reg_bones(x)
            cls_feat = self.cls_head(cls_tower)  # shape=[bs,all_anchor,num_cls]
            reg_feat = self.scales[j](self.reg_head(reg_tower))  # shape=[bs,all_anchor,4*(reg_max+1)]
            reg_score = self.reg_conf(reg_feat)  # shape=[bs,all_anchor,1]

            if cls_feat.dtype == 'float16':
                cls_feat = cls_feat.float()
            if reg_score.dtype == 'float16':
                reg_score = reg_score.float()
            if reg_feat.dtype == 'float16':
                reg_feat = reg_feat.float()

            cls_score = jt.sigmoid(cls_feat) * jt.sigmoid(reg_score)
            cls_outputs.append(cls_score)
            reg_outputs.append(reg_feat)

        if self.anchors[0] is None or self.anchors[0].shape[0] != cls_outputs[0].shape[1]:
            anchors = self.build_anchors(xs)
            assert len(anchors) == len(self.anchors)
            for i, anchor in enumerate(anchors):
                self.anchors[i] = anchor.to(xs[0].device)

        if self.is_training():
            return cls_outputs, reg_outputs, self.anchors
        else:
            predicts_list = list()
            for cls_out, reg_out, stride, anchor in zip(cls_outputs, reg_outputs, self.strides, self.anchors):
                reg_out = self.project(reg_out) * stride  # shape=[bs,all_anchor,4] 4==>ltrb
                anchor_center = ((anchor[:, :2] + anchor[:, 2:]) * 0.5)[None, ...]
                x1y1 = anchor_center - reg_out[..., :2]
                x2y2 = anchor_center + reg_out[..., 2:]
                box_xyxy = jt.concat([x1y1, x2y2], dim=-1)
                predicts_out = jt.concat([box_xyxy, cls_out], dim=-1)  # shape=[bs,all_anchor,4+80]
                predicts_list.append(predicts_out)
            return predicts_list


default_cfg = {
    'num_cls': 80,
    'anchor_sizes': [32., 64., 128., 256., 512.],
    'anchor_scales': [2 ** 0, ],
    'anchor_ratios': [1., ],
    'strides': [8, 16, 32, 64, 128],
    'backbone': 'resnet18',
    'pretrained': True,
    'fpn_channel': 256,
    'head_conv_num': 4,
    'block_type': 'CGR',
    'reg_max': 16,
    'm_top_k': 4,
    'subnet_dim': 64,
    'add_mean': True,
    # loss
    'top_k': 9,
    'iou_loss_weight': 2.0,
    'reg_loss_weight': 0.25,
    'beta': 2.0,
    'iou_type': 'giou',
    # predicts
    'conf_thresh': 0.01,
    'nms_iou_thresh': 0.5,
    'max_det': 300,
}


class GFocal(nn.Module):
    def __init__(self, **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        super(GFocal, self).__init__()
        self.backbones = getattr(resnet, self.cfg['backbone'])(pretrained=self.cfg['pretrained'])
        c3, c4, c5 = self.backbones.inner_channels
        self.neck = FPN(c3, c4, c5, self.cfg['fpn_channel'])
        self.head = GFocalHead(in_channel=self.cfg['fpn_channel'],
                             inner_channel=self.cfg['fpn_channel'],
                             num_cls=self.cfg['num_cls'],
                             num_convs=self.cfg['head_conv_num'],
                             layer_num=5,
                             anchor_sizes=self.cfg['anchor_sizes'],
                             anchor_scales=self.cfg['anchor_scales'],
                             anchor_ratios=self.cfg['anchor_ratios'],
                             strides=self.cfg['strides'],
                             block_type=self.cfg['block_type'],
                             reg_max=self.cfg['reg_max'],
                             subnet_dim=self.cfg['subnet_dim'],
                             m_top_k=self.cfg['m_top_k'],
                             add_mean=self.cfg['add_mean'])
        self.loss = GFocalLoss(
            strides=self.cfg['strides'],
            top_k=self.cfg['top_k'],
            beta=self.cfg['beta'],
            iou_loss_weight=self.cfg['iou_loss_weight'],
            reg_loss_weight=self.cfg['reg_loss_weight'],
            iou_type=self.cfg['iou_type'],
            anchor_num_per_loc=self.head.anchor_nums
        )

    def execute(self, x, targets=None):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])

        ret = dict()
        if self.is_training():
            assert targets is not None
            cls_outputs, reg_outputs, anchors = out
            loss_qfl, loss_iou, loss_dfl, num_pos = self.loss(
                cls_outputs, reg_outputs, anchors, targets)
            ret['loss_qfl'] = loss_qfl
            ret['loss_iou'] = loss_iou
            ret['loss_dfl'] = loss_dfl
            ret['match_num'] = num_pos
        else:
            _, _, h, w = x.shape
            for pred in out:
                pred[:, [0, 2]] = jt.clamp(pred[:, [0, 2]], min_v=0, max_v=w)
                pred[:, [1, 3]] = jt.clamp(pred[:, [1, 3]], min_v=0, max_v=h)
            predicts = non_max_suppression(out,
                                        conf_thresh=self.cfg['conf_thresh'],
                                        iou_thresh=self.cfg['nms_iou_thresh'],
                                        max_det=self.cfg['max_det']
                                        )
            ret['predicts'] = predicts
        return ret


if __name__ == '__main__':
    input_tensor = jt.rand(size=(4, 8525, 68))
    net = Project(reg_max=16)
    x = net(input_tensor)
    print(x.shape) 