import torch
import torch.nn.functional as F
from utils.boxs_utils import box_iou
from losses.commons import BoxSimilarity,IOULoss


INF=1e8



def distance2box(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return torch.stack([x1, y1, x2, y2], -1)


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
    return torch.stack([l, t, r, b], -1)




class Project(object):
    def __init__(self,reg_max=16):
        super(Project, self).__init__()
        self.reg_max=reg_max
        self.project=torch.linspace(0,self.reg_max,self.reg_max+1)

    def __call__(self, x):
        '''
        :param x: shape=[b,n,4*(reg_max+1)]
        :return:
        x: shape=[b,n,4]
        '''
        if self.project.device != x.device:
            self.project=self.project.to(x.device)
        b,n,c=x.shape
        x=x.view(b,-1,self.reg_max+1).softmax(dim=-1)
        x=F.linear(x,self.project).view(b,n,-1)
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
        self.ce=torch.nn.CrossEntropyLoss(reduction='none')

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
                _,topk_idxs_per_level=distances_per_level.topk(top_k,dim=0,largest=False)
                candidate_idxs.append(topk_idxs_per_level+start_idx)
                start_idx+=num_anchor

            candidate_idxs=torch.cat(candidate_idxs,dim=0)
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
            is_in_gts = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0.01

            is_pos = is_pos & is_in_gts

            gt_idx=torch.arange(len(gt), device=anchor_gt_iou.device)[None,:].repeat((len(candidate_idxs),1))
            match=torch.full_like(anchor_gt_iou,fill_value=-INF)
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

    def __call__(self, cls_predicts, reg_predicts, anchors, targets):
        """
        :param cls_predicts:
        :param reg_predicts:
        :param anchors:
        :param targets:
        :return:
        """
        num_anchors_per_level = [len(item) for item in anchors]
        cls_predicts = torch.cat([item for item in cls_predicts], dim=1)
        reg_predicts = torch.cat([item for item in reg_predicts], dim=1)
        if self.expand_strides is None or self.expand_strides.device != cls_predicts.device:
            expand_strides=sum([[i]*len(j) for i,j in zip(self.strides,anchors)],[])
            self.expand_strides = torch.tensor(expand_strides).to(cls_predicts.device)

        all_anchors=torch.cat([item for item in anchors])
        all_anchors_expand = torch.cat([all_anchors, self.expand_strides[:, None]], dim=-1)
        gt_boxes = targets['target'].split(targets['batch_len'])

        matches = self.matcher(all_anchors, gt_boxes, num_anchors_per_level)
        match_bidx=list()
        match_anchor_idx=list()
        match_gt_idx=list()

        for bid,match in matches:
            anchor_idx=(match>=0).nonzero(as_tuple=False).squeeze(-1)
            match_anchor_idx.append(anchor_idx)
            match_gt_idx.append(match[anchor_idx])
            match_bidx.append(bid)

        cls_batch_idx=sum([[i] * len(j) for i, j in zip(match_bidx, match_anchor_idx)], [])
        cls_anchor_idx = torch.cat(match_anchor_idx)
        cls_label_idx = torch.cat([gt_boxes[i][:,0][j].long() for i,j in zip(match_bidx,match_gt_idx)])
        num_pos=len(cls_batch_idx)

        match_expand_anchors=all_anchors_expand[cls_anchor_idx]
        norm_anchor_center=(match_expand_anchors[:, :2]
                              + match_expand_anchors[:, 2:4]) * 0.5 / match_expand_anchors[:, -1:]

        match_reg_pred=reg_predicts[cls_batch_idx,cls_anchor_idx]  # shape=[all_pos,4*(reg_max+1)]
        match_box_ltrb = self.project(match_reg_pred[None, ...])[0]  # shape=[all_pos,4]
        match_norm_box_xyxy = distance2box(norm_anchor_center, match_box_ltrb)


        match_box_targets=torch.cat([gt_boxes[i][:,1:][j] for i,j in zip(match_bidx,match_gt_idx)])
        match_norm_box_targets = match_box_targets / match_expand_anchors[:, -1:]

        iou_scores = self.box_similarity(match_norm_box_xyxy.detach(), match_norm_box_targets)
        cls_targets = torch.zeros_like(cls_predicts)
        cls_targets[cls_batch_idx, cls_anchor_idx, cls_label_idx] = iou_scores

        cls_scores = cls_predicts[cls_batch_idx, cls_anchor_idx].max(dim=-1)[0].detach()
        division_factor = cls_scores.sum()

        loss_qfl = self.qfl(cls_predicts, cls_targets).sum() / division_factor
        loss_iou = (self.iou_loss(match_norm_box_xyxy, match_norm_box_targets) * cls_scores).sum() / division_factor

        match_norm_ltrb_box = box2distance(norm_anchor_center, match_norm_box_targets).clamp(min=0,
                                                                                             max=self.reg_max - 0.1)
        loss_dfl = (self.dfl(match_reg_pred, match_norm_ltrb_box) *
                    cls_scores[:, None].expand(-1, 4).reshape(-1)).sum() / division_factor

        return loss_qfl, self.iou_loss_weight * loss_iou, self.reg_loss_weight * loss_dfl, num_pos











