import jittor as jt
import jittor.nn as nn
from utils.boxs_utils import box_iou
from losses.commons import BoxSimilarity,IOULoss


INF=1e8



def distance2box(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return jt.stack([x1, y1, x2, y2], -1)


def box2distance(points, bbox):
    l = points[:, 0] - bbox[:, 0]
    t = points[:, 1] - bbox[:, 1]
    r = bbox[:, 2] - points[:, 0]
    b = bbox[:, 3] - points[:, 1]
    # if max_dis is not None:
    #     l = l.clamp(min=0, max=max_dis - dt)
    #     t = t.clamp(min=0, max=max_dis - dt)
    #     r = r.clamp(min=0, max=max_dis - dt)
    #     b = b.clamp(min=0, max=max_dis - dt)
    return jt.stack([l, t, r, b], -1)




class Project(object):
    def __init__(self,reg_max=16):
        super(Project, self).__init__()
        self.reg_max=reg_max
        self.project=jt.linspace(0,self.reg_max,self.reg_max+1)

    def __call__(self, x):
        '''
        :param x: shape=[b,n,4*(reg_max+1)]
        :return:
        x: shape=[b,n,4]
        '''
        device = x.device
        if hasattr(self.project, 'device') and self.project.device != device:
            self.project = self.project.to(device)
        b,n,c=x.shape
        x=x.view(b,-1,self.reg_max+1).softmax(dim=-1)
        x=nn.matmul(x, self.project).view(b,n,-1)
        return x





def binary_cross_entropy(predicts,targets,eps=1e-8):
    '''
    :param predicts:
    :param targets:
    :param eps:
    :return:
    '''
    ret = targets * (predicts.clamp(min=eps).log()) + (1 - targets) * ((1 - predicts).clamp(min=eps).log())
    return -ret



class QFL(object):
    def __init__(self,beta=2.0):
        super(QFL, self).__init__()
        self.beta=beta

    def __call__(self, predicts, targets):
        '''
        :param predicts:  shape=[bs,all_anchor,num_cls]
        :param targets:
        :return:
        '''
        loss = binary_cross_entropy(predicts, targets) * ((targets - predicts).abs().pow(self.beta))
        return loss





class DFL(object):
    def __init__(self):
        super(DFL, self).__init__()
        self.ce=nn.CrossEntropyLoss(reduction='none')

    def __call__(self,predicts,targets):
        '''
        :param predicts: [n,4*(reg_max+1)]
        :param targets:  [n,4]
        :return:
        '''
        n,s=targets.shape
        _,c=predicts.shape
        reg_num=c//s
        targets=targets.view(-1)
        predicts=predicts.view(-1,reg_num)

        disl=targets.long()
        disr=disl+1
        wl=disr.float()-targets
        wr=targets-disl.float()
        loss=self.ce(predicts, disl) * wl + self.ce(predicts, disr) * wr
        return loss






class ATSSMatcher(object):
    def __init__(self,top_k,anchor_num_per_loc):
        self.top_k=top_k
        self.anchor_num_per_loc=anchor_num_per_loc

    def __call__(self,anchors, gt_boxes, num_anchor_per_layer):
        '''
        :param anchors:
        :param gt_boxes:
        :param num_anchor_per_layer:
        :return:
        '''

        ret_list=list()
        anchor_xy=(anchors[:,:2]+anchors[:,2:])/2.

        for bid,gt in enumerate(gt_boxes):
            if len(gt) == 0:
                continue
            start_idx = 0
            candidate_idxs = list()
            gt_xy=(gt[:,[1,2]]+gt[:,[3,4]])/2.
            distances=(anchor_xy[:,None,:]-gt_xy[None,:,:]).pow(2).sum(-1).sqrt()  # shape=[all_anchor,num_gt]
            anchor_gt_iou=box_iou(anchors,gt[:,1:])  # shape=[all_anchor,num_gt]
            for num_anchor in num_anchor_per_layer:
                distances_per_level=distances[start_idx:start_idx+num_anchor]
                top_k = min(self.top_k * self.anchor_num_per_loc, num_anchor)
                topk_vals, topk_idxs_per_level=distances_per_level.topk(top_k,dim=0,largest=False)
                candidate_idxs.append(topk_idxs_per_level+start_idx)
                start_idx+=num_anchor

            candidate_idxs=jt.concat(candidate_idxs,dim=0)
            candidate_ious=anchor_gt_iou.gather(dim=0,index=candidate_idxs)  # shape=[sum_topk,num_gt]

            #筛选条件1 iou>统计量
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]  # shape=[sum_topk,num_gt]
            # 筛选条件2 中心点在gt_box内部
            candidate_xy=anchor_xy[candidate_idxs]
            lt=candidate_xy-gt[None,:,[1,2]]
            rb=gt[None,:,[3,4]]-candidate_xy
            is_in_gts = jt.concat([lt, rb], dim=-1).min(-1)[0] > 0.01

            is_pos = is_pos & is_in_gts

            gt_idx=jt.arange(len(gt), dtype=jt.int32)[None,:].repeat((len(candidate_idxs),1))
            match=jt.full_like(anchor_gt_iou,fill_value=-INF)
            match[candidate_idxs[is_pos],gt_idx[is_pos]] = anchor_gt_iou[candidate_idxs[is_pos], gt_idx[is_pos]]
            val, match_gt_idx = match.max(dim=1)
            match_gt_idx[val == -INF] = -1
            ret_list.append((bid, match_gt_idx))
        return ret_list






class GFocalLoss(object):
    def __init__(self,top_k,
                 anchor_num_per_loc,
                 strides,
                 beta=2.,
                 iou_type="giou",
                 iou_loss_weight=2.0,
                 reg_loss_weight=0.25,
                 reg_max=16
                 ):
        self.top_k = top_k
        self.beta = beta
        self.reg_max = reg_max
        self.iou_type = iou_type
        self.iou_loss_weight = iou_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.matcher = ATSSMatcher(self.top_k, anchor_num_per_loc)
        self.iou_loss = IOULoss(iou_type=iou_type)
        self.box_similarity = BoxSimilarity(iou_type="iou")  # given class-iou joint score
        self.strides = strides
        self.expand_strides = None
        self.project = Project(reg_max=reg_max)
        self.qfl = QFL(beta=beta)
        self.dfl = DFL()
        
    def __call__(self, 
                 cls_predicts, 
                 reg_predicts, 
                 iou_predicts, 
                 anchors, 
                 gt_boxes, 
                 num_anchor_per_layer
                 ):
        """
        :param cls_predicts:
        :param reg_predicts:
        :param iou_predicts:
        :param anchors:
        :param gt_boxes:
        :param num_anchor_per_layer:
        :return:
        """
        if self.expand_strides is None:
            # 生成anchor对应的stride: [num_anchor,]
            expand_strides = list()
            for stride, num_anchor in zip(self.strides, num_anchor_per_layer):
                expand_strides.append(jt.full((num_anchor,), stride))
            self.expand_strides = jt.concat(expand_strides, dim=0)
            
        # anchor points 回归起点
        anchor_points = ((anchors[:, :2] + anchors[:, 2:]) / 2).detach()
        
        # 匹配正负样本
        matcher_ret_list = self.matcher(anchors.detach(), gt_boxes, num_anchor_per_layer)
        
        # 统计总的匹配正负样本数
        anchor_num = anchors.shape[0]
        
        # 多个批次的信息
        batch_size = len(gt_boxes)
        cls_targets = jt.zeros_like(cls_predicts)
        iou_targets = jt.zeros_like(iou_predicts)
        # 为了方便处理这里只去掉了iou前的维度 [bs=1, num_gt, 4]->[num_gt, 4]
        reg_targets = jt.zeros((anchor_num, 4), dtype=cls_predicts.dtype)
        
        num_matched_anchor = 0
        for (batch_id, matched_gt_indexes) in matcher_ret_list:
            if (matched_gt_indexes >= 0).sum() == 0:
                continue
            
            # 获取batch_id 对应的gt_boxes信息
            batch_gt_boxes = gt_boxes[batch_id]
            
            # 前景索引
            pos_mask = matched_gt_indexes >= 0 
            
            pos_indexes = jt.where(pos_mask)[0] 
            
            num_matched_anchor += len(pos_indexes)
            
            # 获取匹配的gt索引
            matched_gt_indexes = matched_gt_indexes[pos_mask]
            
            # 获取所有前景点对应的anchors坐标
            pos_anchors = anchors[pos_indexes]
            
            # 获取所有前景点匹配的gt的坐标
            matched_gt = batch_gt_boxes[matched_gt_indexes, 1:] 
            
            # 计算每个前景anchor与对应gt的iou,用于分类分支的权重
            ious = self.box_similarity(pos_anchors, matched_gt).detach()
            
            # 生成分类目标,这里为什么要乘以iou呢
            gt_labels = batch_gt_boxes[matched_gt_indexes, 0].long()
            
            # 我们需要确保索引 batch_id 在 cls_targets 的有效范围内
            if batch_id < cls_targets.shape[0]:  # 检查 batch_id 是否在有效范围内
                if cls_targets.ndim == 3:  # [bs, anchor_num, num_cls] 形状
                    # 使用 one_hot 编码设置目标类别
                    cls_targets[batch_id, pos_indexes, gt_labels] = ious
                else:  # 假设是 [anchor_num, num_cls] 形状
                    cls_targets[pos_indexes, gt_labels] = ious
            
            # 计算回归位置的偏移量
            # 获取anchor points
            pos_anchor_points = anchor_points[pos_indexes]
            
            # 计算位置编码
            pos_strides = self.expand_strides[pos_indexes]
            
            # gt 相对于对应anchor point的距离
            reg_targets[pos_indexes] = box2distance(pos_anchor_points, matched_gt) / pos_strides[:, None]
            
            # 回归分支质量评分
            iou_targets[batch_id, pos_indexes, 0] = ious.detach()
        
        avg_factor = max(1, num_matched_anchor)
        
        # 解码回归分支
        decode_predicts = distance2box(anchor_points, self.project(reg_predicts) * self.expand_strides[:, None])
        
        # 处理无正样本
        if num_matched_anchor <= 0:
            loss_qfl = cls_predicts.sum() * 0
            loss_dfl = reg_predicts.sum() * 0
            loss_iou = iou_predicts.sum() * 0
            return {
                "loss_qfl": loss_qfl,
                "loss_dfl": loss_dfl,
                "loss_iou": loss_iou,
                "match_num": 0
            }
        
        # 独自focal loss
        # foreground
        pos_mask = cls_targets > 0
        # background
        neg_mask = cls_targets == 0
        
        # 前景损失
        qfl_mask = pos_mask | neg_mask
        qfl_mask = jt.cast(qfl_mask, cls_predicts.dtype)
        loss_qfl = self.qfl(cls_predicts, cls_targets) * qfl_mask
        loss_qfl = loss_qfl.sum() / avg_factor
        
        pos_anchor_points = anchor_points.unsqueeze(0).repeat((batch_size, 1, 1))
        pos_strides = self.expand_strides.unsqueeze(0).repeat((batch_size, 1))
        
        # 只计算前景的回归损失
        reg_mask = pos_mask.max(axis=-1, keepdims=True)[0]
        loss_dfl = self.dfl(reg_predicts[reg_mask[:, :, 0]], reg_targets[reg_mask[:, :, 0]])
        loss_dfl = loss_dfl.sum() / avg_factor
        
        loss_iou = self.iou_loss(decode_predicts[reg_mask[:, :, 0]], anchors[reg_mask[:, :, 0]])
        loss_iou = (loss_iou * iou_targets[reg_mask]).sum() / avg_factor
        
        return {
            "loss_qfl": loss_qfl,
            "loss_dfl": loss_dfl * self.reg_loss_weight,
            "loss_iou": loss_iou * self.iou_loss_weight,
            "match_num": num_matched_anchor
        } 