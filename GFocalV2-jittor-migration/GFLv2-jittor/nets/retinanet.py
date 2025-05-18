import jittor as jt
import math
from jittor import nn
from utils.ckpt_load import load_ckpt
from . import resnet
from .common import FPN, CGR, CBR
from patches.torch_adapter import nms

# GFocal默认配置
default_cfg = {
    "fpn_channel": 256,
    "head_conv_num": 4,
    "anchor_scales": [[1.0]],
    "anchor_ratios": [[1.0, 2.0, 0.5]],
    "block_type": "CGR",
    "subnet_dim": 64,
    "m_top_k": 4,
    "add_mean": True
}

def non_max_suppression(prediction: list,
                        conf_thresh=0.05,
                        iou_thresh=0.5,
                        max_det=300,
                        max_box=2048,
                        max_layer_num=1000
                        ):
    """
    非极大值抑制
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
        batch_predicts_list = [jt.zeros((0, 6)).float()] * len(prediction)
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
            batch_predicts = jt.concat([valid_box, valid_val[:, None], valid_cls[:, None]], dim=-1)
            batch_predicts_list[lj] = batch_predicts
        x = jt.concat(batch_predicts_list, dim=0)
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * max_box
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thresh)
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
        self.bones=list()
        for i in range(num):
            if i==0:
                block=eval(block_type)(in_channel,inner_channel,kernel_size,stride,padding=padding,bias=bias)
            else:
                block=eval(block_type)(inner_channel,inner_channel,kernel_size,stride,padding=padding,bias=bias)
            self.bones.append(block)
        self.bones=nn.Sequential(*self.bones)

        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.trunc_normal_(m.weight, std=0.01)
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
        self.num_anchors=num_anchors
        self.num_cls=num_cls
        self.cls=nn.Conv(in_channel, self.num_anchors*self.num_cls, 3, 1, 1)
        nn.init.trunc_normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def execute(self, x):
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, self.num_cls).view(bs, -1, self.num_cls)
        return x






class GFocalRegHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9, reg_max=16):
        super(GFocalRegHead, self).__init__()
        self.num_anchors=num_anchors
        self.num_output=4*(reg_max+1)
        self.reg_max=reg_max
        self.reg=nn.Conv(in_channel, self.num_anchors*self.num_output, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.trunc_normal_(m.weight, std=0.01)
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
        self.m_top_k=m_top_k
        self.add_mean=add_mean
        total_dim=(m_top_k+1)*4 if add_mean else m_top_k*4
        self.reg_conf=nn.Sequential(
            nn.Linear(total_dim, inner_channel),
            nn.ReLU(),
            nn.Linear(inner_channel, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
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
        prob_topk, _ = x.softmax(dim=-1).topk(self.m_top_k, dim=-1)  # shape=[bs,n,4,topk]
        if self.add_mean:
            stat = jt.concat([prob_topk, prob_topk.mean(dim=-1, keepdims=True)], dim=-1)
        else:
            stat = prob_topk
        if stat.dtype != origin_type:
            stat = stat.astype(origin_type)
        quality_score = self.reg_conf(stat.reshape(bs, n, -1))
        return quality_score


class Project(nn.Module):
    def __init__(self, reg_max=16):
        super(Project, self).__init__()
        self.reg_max = reg_max
        self.register_buffer("project", self._project())

    def _project(self):
        """
        生成预测偏移量的系数矩阵
        :return:
        """
        linspace = jt.linspace(0, self.reg_max, self.reg_max + 1)
        return linspace

    def execute(self, x):
        """
        :param x: 预测的回归参数 [bs,n,4*(reg_max+1)]
        :return:
        """
        x = x.reshape(-1, self.reg_max + 1)
        # x = F.softmax(x, dim=1)
        x = x.softmax(dim=1)
        # ret = torch.matmul(x, self.project.to(x.dtype).unsqueeze(-1)).squeeze(-1).reshape(-1, 4)
        ret = jt.matmul(x, self.project.astype(x.dtype).unsqueeze(-1)).squeeze(-1).reshape(-1, 4)
        return ret




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
        self.strides = strides
        self.anchor_sizes = anchor_sizes
        self.layer_num = layer_num
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_nums = len(anchor_scales[0]) * len(anchor_ratios[0])
        self.fpn_stride = 8
        self.reg_max = reg_max

        self.cls_subnet = nn.ModuleList([
            SequenceCNR(in_channel, inner_channel, 3, 1,
                      num=num_convs, padding=None, bias=True, block_type=block_type) for _ in range(self.layer_num)
        ])
        self.reg_subnet = nn.ModuleList([
            SequenceCNR(in_channel, inner_channel, 3, 1,
                      num=num_convs, padding=None, bias=True, block_type=block_type) for _ in range(self.layer_num)
        ])
        self.cls_score_net = nn.ModuleList([
            GFocalClsHead(inner_channel, self.anchor_nums, num_cls) for _ in range(self.layer_num)
        ])
        self.reg_pred_net = nn.ModuleList([
            GFocalRegHead(inner_channel, self.anchor_nums, reg_max) for _ in range(self.layer_num)
        ])

        self.scales = nn.ModuleList([Scale() for _ in range(self.layer_num)])
        self.project = Project(reg_max)
        self.quality_score_net = SubNetFC(m_top_k, subnet_dim, add_mean)
        self.anchors = None

    def build_anchors_delta(self, size=32.):
        """
        生成一个网格的9个anchors
        :param size:
        :return:
        """
        scales = jt.array(self.anchor_scales[0])
        ratio = jt.array(self.anchor_ratios[0])
        scale_size = (scales * size)
        w = (scale_size.reshape(-1, 1) * ratio.sqrt().reshape(1, -1)).reshape(-1, 1)
        h = (scale_size.reshape(-1, 1) / ratio.sqrt().reshape(1, -1)).reshape(-1, 1)
        delta = jt.concat([-w, -h, w, h], dim=1) * 0.5
        return delta.view(1, -1, 4)

    def build_anchors(self, feature_maps):
        """
        实际上就是生成每一个网格对应的anchors
        :param feature_maps:
        :return:
        """
        anchors = []
        for i, size in enumerate(self.anchor_sizes):
            fm = feature_maps[i]
            fm_h, fm_w = fm.shape[-2], fm.shape[-1]
            fm_stride = self.strides[i]
            anchor_delta = self.build_anchors_delta(size)
            shift_x = (jt.arange(0, fm_w) + 0.5) * fm_stride
            shift_y = (jt.arange(0, fm_h) + 0.5) * fm_stride
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            xy = jt.stack((shift_x, shift_y), dim=-1).float()
            xy = xy.unsqueeze(1).expand((fm_h * fm_w, self.anchor_nums, 2))
            wh = anchor_delta.expand((fm_h * fm_w, self.anchor_nums, 4))
            xy = xy.reshape(-1, 2)
            wh = wh.reshape(-1, 4)
            xy1 = xy.unsqueeze(1)
            xy2 = xy.unsqueeze(1)
            anchor = jt.concat((xy1, xy2), dim=1) + wh
            anchors.append(anchor)
        return anchors

    def execute(self, xs):
        """
        :param xs:
        :return:
        """
        cls_scores = list()
        reg_preds = list()
        anchors = list()

        self.anchors = self.build_anchors(xs)
        for i, (x, scale, cls_tower, reg_tower, cls_pred, reg_pred) in enumerate(
                zip(xs, self.scales, self.cls_subnet, self.reg_subnet, self.cls_score_net, self.reg_pred_net)):
            cls_feat = cls_tower(x)
            reg_feat = reg_tower(x)
            cls_score = cls_pred(cls_feat)
            reg_pred = reg_pred(reg_feat)
            reg_conf = self.quality_score_net(reg_pred)
            cls_score = jt.sqrt(cls_score.sigmoid() * reg_conf.sigmoid())
            reg_dist = reg_pred.reshape(reg_pred.shape[0], -1, 4, self.reg_max + 1)
            reg_dist = reg_dist.softmax(dim=-1)
            reg_pred = self.project(reg_pred)
            reg_pred = scale(reg_pred)
            cls_scores.append(cls_score)
            reg_preds.append(reg_pred)
            anchors.append(self.anchors[i])

        if not self.training:
            predicts = list()
            for i, (cls_score, reg_pred, anchor) in enumerate(zip(cls_scores, reg_preds, anchors)):
                # 这里不要忘了还原大小
                reg_pred = reg_pred * self.strides[i]
                # 还原成中心点宽高形式
                """
                anchor: x1,y1,x2,y2
                pred_reg : dx,dy,dw,dh
                """
                x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                w = x2 - x1
                h = y2 - y1
                dx, dy, dw, dh = reg_pred[:, 0], reg_pred[:, 1], reg_pred[:, 2], reg_pred[:, 3]
                new_cx = cx + dx
                new_cy = cy + dy
                x1 = new_cx - (w + dw) * 0.5
                y1 = new_cy - (h + dh) * 0.5
                x2 = new_cx + (w + dw) * 0.5
                y2 = new_cy + (h + dh) * 0.5
                box = jt.stack([x1, y1, x2, y2], dim=-1)
                # bs ,grid, box+cls
                full_reg_pred = jt.concat([box, cls_score], dim=-1)
                predicts.append(full_reg_pred)
            return predicts
        return [cls_scores, reg_preds, anchors]




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
        
        # 加载预训练权重(使用backbones中的load方法)
        if self.cfg['pretrained'] and hasattr(self.backbones, 'pretrained') and self.backbones.pretrained:
            self.backbones.load(self.backbones.pretrained)


    def execute(self, x, targets=None):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])

        ret = dict()
        if self.training:
            assert targets is not None
            cls_outputs, reg_outputs, anchors = out
            from losses.gfocal import GFocalLoss
            loss = GFocalLoss(
                strides=self.cfg['strides'],
                top_k=self.cfg['top_k'],
                beta=self.cfg['beta'],
                iou_loss_weight=self.cfg['iou_loss_weight'],
                reg_loss_weight=self.cfg['reg_loss_weight'],
                iou_type=self.cfg['iou_type'],
                anchor_num_per_loc=self.head.anchor_nums
            )
            loss_qfl, loss_iou, loss_dfl, num_pos = loss(
                cls_outputs, reg_outputs, anchors, targets)
            ret['loss_qfl'] = loss_qfl
            ret['loss_iou'] = loss_iou
            ret['loss_dfl'] = loss_dfl
            ret['match_num'] = num_pos
        else:
            _, _, h, w = x.shape
            for pred in out:
                pred[:, [0, 2]] = pred[:, [0, 2]].clamp(min=0, max=w)
                pred[:, [1, 3]] = pred[:, [1, 3]].clamp(min=0, max=h)
            predicts = non_max_suppression(out,
                                           conf_thresh=self.cfg['conf_thresh'],
                                           iou_thresh=self.cfg['nms_iou_thresh'],
                                           max_det=self.cfg['max_det']
                                           )
            ret['predicts'] = predicts
        return ret 