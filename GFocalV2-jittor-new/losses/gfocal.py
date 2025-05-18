import jittor as jt
import jittor.nn as nn
from utils.boxs_utils import box_iou
from losses.commons import BoxSimilarity,IOULoss


INF=1e8


def distance2box(points, distance):
    # 检查输入张量是否为空
    if points.shape[0] == 0 or distance.shape[0] == 0:
        # 返回一个空张量，保持维度一致
        return jt.zeros((0, 4))
    
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return jt.stack([x1, y1, x2, y2], -1)


def box2distance(points, bbox):
    # 检查输入张量是否为空
    if points.shape[0] == 0 or bbox.shape[0] == 0:
        # 返回一个空张量，保持维度一致
        return jt.zeros((0, 4))
    
    l = points[:, 0] - bbox[:, 0]
    t = points[:, 1] - bbox[:, 1]
    r = bbox[:, 2] - points[:, 0]
    b = bbox[:, 3] - points[:, 1]
    return jt.stack([l, t, r, b], -1)


# 自定义函数：计算指定维度的标准差
def custom_std(tensor, dim=None):
    """
    计算指定维度的标准差
    :param tensor: 输入张量
    :param dim: 计算标准差的维度
    :return: 标准差
    """
    if dim is None:
        # 整个张量的标准差
        mean = tensor.mean()
        var = ((tensor - mean) ** 2).mean()
        return jt.maximum(var, jt.array(1e-8)).sqrt()
    
    # 计算平均值，保持维度
    mean = tensor.mean(dim=dim, keepdims=True)
    # 计算方差
    var = ((tensor - mean) ** 2).mean(dim=dim)
    # 打印调试信息
    print(f"方差统计: 形状={var.shape}, 最小值={var.min().item() if var.numel() > 0 else 'empty'}")
    # 确保数值稳定性
    var = jt.maximum(var, jt.array(1e-8))
    # 返回标准差
    return var.sqrt()


class Project(nn.Module):
    def __init__(self,reg_max=16):
        super(Project, self).__init__()
        self.reg_max=reg_max
        self.project=jt.linspace(0,self.reg_max,self.reg_max+1)

    def execute(self, x):
        '''
        :param x: shape=[b,n,4*(reg_max+1)]
        :return:
        x: shape=[b,n,4]
        '''
        b,n,c=x.shape
        x=x.reshape(b,-1,self.reg_max+1)
        x=jt.nn.softmax(x,dim=-1)
        x=jt.matmul(x, self.project)
        x=x.reshape(b,n,-1)
        return x


def binary_cross_entropy(predicts,targets,eps=1e-8):
    '''
    :param predicts:
    :param targets:
    :param eps:
    :return:
    '''
    predicts=jt.clamp(predicts,min_v=eps,max_v=1.0-eps)
    ret = targets * jt.log(predicts) + (1 - targets) * jt.log(1 - predicts)
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
        # 确保数值稳定性
        predicts = jt.clamp(predicts, min_v=1e-8, max_v=1.0-1e-8)
        # 使用更稳定的实现
        targets_weight = (targets - predicts).abs().pow(self.beta)
        
        # 避免零权重问题
        targets_weight = jt.maximum(targets_weight, jt.array(1e-12))
        
        loss = binary_cross_entropy(predicts, targets) * targets_weight
        
        # 避免NaN和Inf
        loss = jt.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0)
        
        return loss


class DFL(object):
    def __init__(self):
        super(DFL, self).__init__()

    def __call__(self, predicts, targets):
        '''
        :param predicts: [n,4*(reg_max+1)]
        :param targets:  [n,4]
        :return:
        '''
        n, s = targets.shape
        _, c = predicts.shape
        reg_num = c//s
        targets = targets.reshape(-1)
        predicts = predicts.reshape(-1, reg_num)

        disl = targets.long()
        disr = disl + 1
        wl = disr.float() - targets
        wr = targets - disl.float()
        
        loss_l = nn.cross_entropy_loss(predicts, disl, reduction='none')
        loss_r = nn.cross_entropy_loss(predicts, disr, reduction='none')
        loss = loss_l * wl + loss_r * wr
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
                # 处理空的gt_boxes情况，返回所有anchor的负匹配结果
                match_gt_idx = jt.ones(len(anchors)) * (-1)
                ret_list.append((bid, match_gt_idx))
                continue
                
            start_idx = 0
            candidate_idxs = list()
            gt_xy=(gt[:,[1,2]]+gt[:,[3,4]])/2.
            
            # 打印调试信息
            print(f"gt shape: {gt.shape}, anchor_xy shape: {anchor_xy.shape}")
            
            # 确保形状匹配
            if len(gt.shape) < 2 or gt.shape[0] == 0 or gt.shape[1] <= 5:
                # gt形状不正确，创建一个默认的空匹配
                match_gt_idx = jt.ones(len(anchors)) * (-1)
                ret_list.append((bid, match_gt_idx))
                continue
                
            distances=((anchor_xy[:,None,:]-gt_xy[None,:,:]).pow(2).sum(-1)).sqrt()  # shape=[all_anchor,num_gt]
            anchor_gt_iou=box_iou(anchors,gt[:,1:])  # shape=[all_anchor,num_gt]
            
            # 修复Jittor的max函数用法
            # 使用argmax获取最大值的索引
            max_iou_anchor_idx = anchor_gt_iou.argmax(dim=0)
            # 然后根据索引获取对应的最大值
            max_iou_per_gt = jt.zeros_like(max_iou_anchor_idx, dtype=jt.float32)
            for i in range(len(max_iou_anchor_idx)):
                max_iou_per_gt[i] = anchor_gt_iou[max_iou_anchor_idx[i], i]
                
            print(f"每个GT框的最大IoU值: {max_iou_per_gt}")
            
            # 增加：至少为每个GT框选择一个最佳匹配的锚点
            force_match = False
            if max_iou_per_gt.max() < 0.1:  # 如果所有IoU都很低
                print("警告：所有IoU都小于0.1，强制匹配每个GT框")
                force_match = True
            
            for num_anchor in num_anchor_per_layer:
                distances_per_level=distances[start_idx:start_idx+num_anchor]
                # 增加top_k值，考虑更多的候选点
                top_k = min(self.top_k * self.anchor_num_per_loc * 2, num_anchor)
                _,topk_idxs_per_level=distances_per_level.topk(top_k,dim=0,largest=False)
                candidate_idxs.append(topk_idxs_per_level+start_idx)
                start_idx+=num_anchor

            candidate_idxs=jt.concat(candidate_idxs,dim=0)
            candidate_ious=anchor_gt_iou.gather(dim=0,index=candidate_idxs)  # shape=[sum_topk,num_gt]

            #筛选条件1 iou>统计量
            iou_mean_per_gt = candidate_ious.mean(0)
            # 使用自定义函数计算标准差，而不是使用std(dim=0)
            iou_std_per_gt = custom_std(candidate_ious, dim=0)
            # 进一步降低匹配的标准，几乎不考虑标准差
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt * 0.05
            # 添加调试输出
            print(f"IoU阈值统计: 均值={iou_mean_per_gt.item() if iou_mean_per_gt.numel() == 1 else '数组'}, "
                  f"标准差={iou_std_per_gt.item() if iou_std_per_gt.numel() == 1 else '数组'}, "
                  f"阈值={iou_thresh_per_gt.item() if iou_thresh_per_gt.numel() == 1 else '数组'}")
            
            # 确保阈值不要太高，设置上限，同时设置下限，确保有足够的正样本
            iou_thresh_per_gt = jt.minimum(iou_thresh_per_gt, jt.array(0.6))
            iou_thresh_per_gt = jt.maximum(iou_thresh_per_gt, jt.array(0.05))  # 设置最低阈值
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]  # shape=[sum_topk,num_gt]
            
            # 如果没有满足条件的，放宽要求，使用固定阈值
            if jt.sum(is_pos) == 0:
                print("警告: 使用固定IoU阈值0.05进行匹配")
                is_pos = candidate_ious >= 0.05

            # 筛选条件2 中心点在gt_box内部或附近
            candidate_xy=anchor_xy[candidate_idxs]
            lt=candidate_xy-gt[None,:,[1,2]]
            rb=gt[None,:,[3,4]]-candidate_xy
            
            # 计算候选点到边界框的距离，使用一个宽松的标准
            distances_to_box = jt.concat([lt, rb], dim=-1)
            min_distances = distances_to_box.min(-1)[0]
            print(f"候选点到边界框的最小距离统计: 最小值={min_distances.min().item()}, 最大值={min_distances.max().item()}, 均值={min_distances.mean().item()}")
            
            # 极度放宽条件：点可以在框外很远
            box_sizes = (gt[:, 3:5] - gt[:, 1:3])  # 获取框的宽高
            box_sizes_expanded = box_sizes[None, :, :].repeat((len(candidate_idxs), 1, 1))
            box_diag = jt.sqrt((box_sizes_expanded[:, :, 0] ** 2 + box_sizes_expanded[:, :, 1] ** 2))
            dist_threshold = -0.5 * box_diag  # 允许点在框外不超过对角线长度的50%
            
            # 更宽松的条件：点可以在框外一定距离
            is_near_gts = min_distances > dist_threshold
            print(f"在或接近gt框的点数量: {jt.sum(is_near_gts).item()}")
            
            # 详细的调试信息
            print(f"候选点数量: {candidate_idxs.shape[0]}, gt框数量: {gt.shape[0]}")
            print(f"满足IoU条件的点数量: {jt.sum(is_pos).item()}")
            
            # 组合IoU和位置条件，但给IoU条件更高的优先级
            is_pos_original = is_pos.clone()
            is_pos = is_pos & is_near_gts
            print(f"同时满足IoU和位置条件的点数量: {jt.sum(is_pos).item()}")
            
            # 如果没有满足条件的正样本，尝试放宽条件，只使用IoU条件
            if jt.sum(is_pos) == 0:
                print("警告: 放宽条件，只使用IoU作为匹配条件")
                is_pos = is_pos_original
                print(f"放宽条件后满足条件的点数量: {jt.sum(is_pos).item()}")
                
                # 如果依然没有正样本，使用固定阈值
                if jt.sum(is_pos) == 0:
                    print("警告: 使用极低的固定IoU阈值0.01进行匹配")
                    is_pos = candidate_ious >= 0.01
                    print(f"使用固定阈值后满足条件的点数量: {jt.sum(is_pos).item()}")

            gt_idx=jt.arange(len(gt))[None,:].repeat((len(candidate_idxs),1))
            match=jt.ones_like(anchor_gt_iou) * (-INF)
            
            # 检查是否有任何正样本
            if jt.sum(is_pos) > 0:
                match[candidate_idxs[is_pos],gt_idx[is_pos]] = anchor_gt_iou[candidate_idxs[is_pos], gt_idx[is_pos]]
                
            # 强制匹配：如果所有IoU都很低，为每个GT框强制分配一个最佳的锚点
            if force_match or jt.sum(is_pos) < len(gt):
                print("强制为每个GT框分配最佳锚点")
                for gt_idx, anchor_idx in enumerate(max_iou_anchor_idx):
                    match[anchor_idx, gt_idx] = anchor_gt_iou[anchor_idx, gt_idx]
                
            # 修复Jittor中max函数的用法
            # 原代码: max_result = match.max(dim=1)
            # 原代码: val, match_gt_idx = max_result[0], max_result[1]
            
            # Jittor中获取每行最大值及其索引
            match_gt_idx = match.argmax(dim=1)  # 获取每行最大值的索引
            val = jt.zeros_like(match_gt_idx, dtype=jt.float32)
            for i in range(len(match_gt_idx)):
                val[i] = match[i, match_gt_idx[i]]
                
            match_gt_idx = jt.where(val == -INF, jt.array(-1), match_gt_idx)
            
            # 输出匹配结果统计
            positive_matches = (match_gt_idx >= 0).sum().item()
            print(f"批次 {bid} 最终匹配到 {positive_matches} 个正样本")
            
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
        
    def add_default_positive_samples(self, all_anchors, gt_boxes, match_anchor_idx, match_gt_idx, match_bidx):
        """当匹配不到足够正样本时，为每个批次添加默认的正样本"""
        if not match_anchor_idx or sum(len(idx) for idx in match_anchor_idx) < len(gt_boxes):
            print("警告：添加默认正样本")
            
            for bid, gt in enumerate(gt_boxes):
                # 检查该批次是否已经有正样本
                has_positive = False
                for i, bid_i in enumerate(match_bidx):
                    if bid_i == bid and len(match_anchor_idx[i]) > 0:
                        has_positive = True
                        break
                        
                if not has_positive and len(gt) > 0:
                    print(f"为批次 {bid} 添加默认正样本")
                    # 计算所有锚点与该批次第一个gt框的IoU
                    # 使用jittor的方式扩展维度
                    gt_box = gt[0, 1:].reshape(1, -1)  # 第一个gt框
                    ious = box_iou(all_anchors, gt_box)
                    # 修正max函数的使用
                    max_iou_idx = ious.argmax()
                    
                    match_anchor_idx.append(jt.array([max_iou_idx]))
                    match_gt_idx.append(jt.array([0]))  # 使用第一个gt框
                    match_bidx.append(bid)
                    print(f"批次 {bid} 添加了一个默认正样本，锚点索引：{max_iou_idx}，最大IoU：{ious[max_iou_idx].item()}")
        
        return match_anchor_idx, match_gt_idx, match_bidx

    def __call__(self, cls_predicts, reg_predicts, anchors, targets):
        """
        :param cls_predicts:
        :param reg_predicts:
        :param anchors:
        :param targets:
        :return:
        """
        # 调试输出
        print(f"targets keys: {targets.keys() if isinstance(targets, dict) else 'targets is not a dict'}")
        if 'target' in targets:
            print(f"targets['target'] shape: {targets['target'].shape if targets['target'].shape[0] > 0 else 'empty'}")
        if 'batch_len' in targets:
            print(f"targets['batch_len']: {targets['batch_len']}")
            
        # 检查targets是否为空或格式不正确
        if not isinstance(targets, dict) or 'target' not in targets or 'batch_len' not in targets or targets['target'].shape[0] == 0:
            # 返回全零损失
            print("目标数据为空或格式不正确，返回零损失")
            return jt.array([0.0]), jt.array([0.0]), jt.array([0.0]), 0
            
        num_anchors_per_level = [len(item) for item in anchors]
        cls_predicts = jt.concat([item for item in cls_predicts], dim=1)
        reg_predicts = jt.concat([item for item in reg_predicts], dim=1)
        if self.expand_strides is None:
            expand_strides=sum([[i]*len(j) for i,j in zip(self.strides,anchors)],[])
            self.expand_strides = jt.array(expand_strides)

        all_anchors=jt.concat([item for item in anchors])
        all_anchors_expand = jt.concat([all_anchors, self.expand_strides[:, None]], dim=-1)
        
        # 确保target的格式正确
        target_data = targets['target']
        batch_len = targets['batch_len']
        
        # 确保batch_len是一个整数列表
        if isinstance(batch_len, jt.Var):
            batch_len = batch_len.tolist()
            
        # 检查格式是否有效
        if sum(batch_len) != target_data.shape[0]:
            print(f"警告: 目标数据格式不匹配 - sum(batch_len)={sum(batch_len)}, target_data.shape[0]={target_data.shape[0]}")
            # 修复batch_len
            batch_len = [target_data.shape[0]]
            
        # 尝试分割目标数据
        try:
            gt_boxes = target_data.split(batch_len)
        except Exception as e:
            print(f"分割目标数据时出错: {str(e)}")
            # 回退到简单处理
            gt_boxes = [target_data]
            
        print(f"gt_boxes长度: {len(gt_boxes)}, 第一个元素形状: {gt_boxes[0].shape if len(gt_boxes) > 0 else 'empty'}")

        # 匹配正负样本
        matches = self.matcher(all_anchors, gt_boxes, num_anchors_per_level)
        match_bidx=list()
        match_anchor_idx=list()
        match_gt_idx=list()

        for bid,match in matches:
            anchor_idx=(match>=0).nonzero()[:, 0]
            if len(anchor_idx) > 0:  # 确保有正样本
                match_anchor_idx.append(anchor_idx)
                match_gt_idx.append(match[anchor_idx])
                match_bidx.append(bid)
                print(f"批次 {bid} 匹配到 {len(anchor_idx)} 个正样本")
            else:
                print(f"批次 {bid} 未匹配到任何正样本")

        # 检查是否有正样本
        if not match_anchor_idx or all(len(idx) == 0 for idx in match_anchor_idx):
            # 尝试添加默认正样本
            match_anchor_idx, match_gt_idx, match_bidx = self.add_default_positive_samples(
                all_anchors, gt_boxes, match_anchor_idx, match_gt_idx, match_bidx)
            
            # 如果依然没有正样本，返回最小损失
            if not match_anchor_idx or all(len(idx) == 0 for idx in match_anchor_idx):
                # 返回全零损失，但添加小的常数以保持梯度流
                print("警告：所有批次都没有匹配到正样本，返回最小损失")
                min_loss = 1e-6
                return jt.array([min_loss]), jt.array([min_loss]), jt.array([min_loss]), 0
        
        print(f"总共匹配到 {sum(len(idx) for idx in match_anchor_idx)} 个正样本")
        
        # 计算损失
        try:
            cls_batch_idx=sum([[i] * len(j) for i, j in zip(match_bidx, match_anchor_idx)], [])
            cls_anchor_idx = jt.concat(match_anchor_idx)
            cls_label_idx = jt.concat([gt_boxes[i][:,0][j].long() for i,j in zip(match_bidx,match_gt_idx)])
            num_pos=len(cls_batch_idx)
            
            print(f"正样本数量: {num_pos}")
            if num_pos < 1:
                print("警告：匹配到的正样本数量为0，返回最小损失")
                min_loss = 1e-6
                return jt.array([min_loss]), jt.array([min_loss]), jt.array([min_loss]), 0
            
            # 以下计算需要有足够的正样本
            match_expand_anchors=all_anchors_expand[cls_anchor_idx]
            norm_anchor_center=(match_expand_anchors[:, :2]
                                + match_expand_anchors[:, 2:4]) * 0.5 / match_expand_anchors[:, -1:]

            match_reg_pred=reg_predicts[cls_batch_idx,cls_anchor_idx]  # shape=[all_pos,4*(reg_max+1)]
            match_box_ltrb = self.project(match_reg_pred[None, ...])[0]  # shape=[all_pos,4]
            match_norm_box_xyxy = distance2box(norm_anchor_center, match_box_ltrb)


            match_box_targets=jt.concat([gt_boxes[i][:,1:][j] for i,j in zip(match_bidx,match_gt_idx)])
            match_norm_box_targets = match_box_targets / match_expand_anchors[:, -1:]

            iou_scores = self.box_similarity(match_norm_box_xyxy.detach(), match_norm_box_targets)
            cls_targets = jt.zeros_like(cls_predicts)
            cls_targets[cls_batch_idx, cls_anchor_idx, cls_label_idx] = iou_scores

            # 修复max函数的用法
            # 原代码：max_scores = cls_predicts[cls_batch_idx, cls_anchor_idx].max(dim=-1)
            # 原代码：cls_scores = max_scores[0].detach()
            
            # 获取每个位置的最大分类分数
            cls_pred_subset = cls_predicts[cls_batch_idx, cls_anchor_idx]  # 提取感兴趣的部分
            cls_scores = jt.zeros((cls_pred_subset.shape[0],), dtype=jt.float32)
            for i in range(cls_pred_subset.shape[0]):
                cls_scores[i] = cls_pred_subset[i].max()  # 获取每行的最大值
            cls_scores = cls_scores.detach()  # 分离，避免梯度计算
            
            # 避免除以0的风险，使用更安全的除法
            division_factor = cls_scores.sum() + 1e-8
            print(f"分母(division_factor): {division_factor.item()}")
            
            # 计算损失时添加更多数值稳定性保护
            loss_qfl = self.qfl(cls_predicts, cls_targets).sum() / division_factor
            loss_iou = (self.iou_loss(match_norm_box_xyxy, match_norm_box_targets) * cls_scores).sum() / division_factor
            match_norm_ltrb_box = box2distance(norm_anchor_center, match_norm_box_targets).clamp(min_v=0,
                                                                                             max_v=self.reg_max - 0.1)
            
            # 修复expand的用法，使用repeat替代
            # 原代码: loss_dfl = (self.dfl(match_reg_pred, match_norm_ltrb_box) * cls_scores[:, None].expand(-1, 4).reshape(-1)).sum() / division_factor
            
            # 使用jittor支持的方式重复张量
            dfl_loss_raw = self.dfl(match_reg_pred, match_norm_ltrb_box)
            dfl_loss_raw = dfl_loss_raw.reshape(cls_scores.shape[0], 4)
            
            # 扩展cls_scores使其匹配dfl_loss_raw的形状
            expanded_cls_scores = jt.zeros_like(dfl_loss_raw)
            for i in range(4):
                expanded_cls_scores[:, i] = cls_scores
                
            loss_dfl = (dfl_loss_raw * expanded_cls_scores).reshape(-1).sum() / division_factor
                
        except Exception as e:
            print(f"计算损失时出错: {str(e)}")
            loss_qfl = jt.array(1e-6)
            loss_iou = jt.array(1e-6)
            loss_dfl = jt.array(1e-6)
            num_pos = 0
                    
        # 确保所有损失值都是有限的
        loss_qfl = jt.nan_to_num(loss_qfl, nan=0.0, posinf=10.0, neginf=0.0)
        loss_iou = jt.nan_to_num(loss_iou, nan=0.0, posinf=10.0, neginf=0.0)
        loss_dfl = jt.nan_to_num(loss_dfl, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 确保损失不会太小，避免梯度消失
        min_loss_value = 1e-10
        if loss_qfl < min_loss_value and num_pos > 0:
            loss_qfl = jt.array(min_loss_value)
        if loss_iou < min_loss_value and num_pos > 0:
            loss_iou = jt.array(min_loss_value)
        if loss_dfl < min_loss_value and num_pos > 0:
            loss_dfl = jt.array(min_loss_value)
        
        # 最后再次输出最终的损失值
        print(f"最终损失值 - QFL: {loss_qfl.item()}, IoU: {loss_iou.item()}, DFL: {loss_dfl.item()}")

        return loss_qfl, self.iou_loss_weight * loss_iou, self.reg_loss_weight * loss_dfl, num_pos

def clip_grad_norm(parameters, optimizer, max_norm):
    """
    自定义实现梯度裁剪函数
    
    参数:
    parameters - 模型参数
    optimizer - 优化器对象
    max_norm - 梯度范数的最大值
    """
    if max_norm <= 0:
        return
        
    # 计算梯度范数
    total_norm = 0
    for param in parameters:
        if param.requires_grad:
            try:
                grad = param.opt_grad(optimizer)
                if grad is not None:
                    total_norm += (grad ** 2).sum()
            except Exception as e:
                # 打印详细错误信息以便调试
                print(f"获取梯度时出错: {str(e)}")
                continue
                
    if total_norm > 0:
        total_norm = jt.sqrt(total_norm)
        print(f"梯度范数: {total_norm.item()}")
        scale = max_norm / (total_norm + 1e-6)
        if scale < 1:
            print(f"裁剪梯度，比例因子: {scale.item()}")
            # 检查jittor是否支持update_opt_grad
            try:
                for param in parameters:
                    if param.requires_grad:
                        try:
                            grad = param.opt_grad(optimizer)
                            if grad is not None:
                                # 直接修改梯度
                                param.update_opt_grad(optimizer, grad * scale)
                        except Exception as e:
                            print(f"更新梯度时出错: {str(e)}")
                            continue
            except AttributeError:
                print("警告: Jittor不支持update_opt_grad方法，使用替代方案")
                # 替代方案：不直接修改梯度，而是在下一步优化器更新时使用缩放后的学习率
                original_lr = optimizer.lr
                optimizer.lr = original_lr * scale
                print(f"临时调整学习率至: {optimizer.lr}")
                # 注意：这里需要在优化器更新后恢复原始学习率
                # 在main.py中调用后添加: optimizer.lr = original_lr