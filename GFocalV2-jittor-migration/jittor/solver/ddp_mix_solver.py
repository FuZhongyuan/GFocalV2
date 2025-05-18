import os
import jittor as jt
import time
from copy import deepcopy
from torch.cuda import amp  # 暂时保留用于scaler功能，之后可能需要自己实现
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datasets.coco import COCO_CLASSES, COCO_NAMES, coco_collate
from utils.boxs_utils import non_max_suppression
from utils.model_utils import ModelEMA, AverageLogger
from metrics.map import coco_map
from utils.model_utils import rand_seed, is_parallel
from utils.optims_utils import split_optimizer
from utils.optims_utils import IterWarmUpCosineDecayMultiStepLRAdjust
from utils.optims_utils import EpochWarmUpCosineDecayLRAdjust


class DDPMixSolver(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if jt.has_cuda else "cpu"
        self.rank = cfg.rank
        self.start_epoch = self.cfg.schedule.start_epoch
        self.epochs = self.cfg.schedule.epochs
        self.use_ema = self.cfg.model.use_ema
        self.use_epp = self.cfg.train.use_epp
        self.auto_resume = self.cfg.model.resume
        self.local_rank = self.cfg.local_rank
        self.img_size = self.cfg.train.img_size
        self.no_val = self.cfg.train.no_val
        self.ema_decay = self.cfg.model.ema_decay
        
        # 设置随机数种子
        rand_seed(self.cfg.seed)
        
        # 数据集设置
        self.img_trains = self.cfg.train.train_img_folder
        self.img_vals = self.cfg.train.val_img_folder
        self.train_ann = self.cfg.train.train_ann
        self.val_ann = self.cfg.train.val_ann
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 创建日志目录
        self.save_prefix = cfg.train.save_prefix
        self.save_dir = os.path.join("./", "checkpoints", self.save_prefix)
        if self.rank in [-1, 0]:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            self.writer = SummaryWriter(self.save_dir)
            self.res_writer = SummaryWriter(os.path.join(self.save_dir, "20cls"))
                
        # 模型初始化
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.ema = None
        self.lr_adjuster = None
        
        # 混合精度设置
        self.use_amp = cfg.train.use_amp
        self.scaler = amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        
        # 设置日志打印频率
        self.print_freq = cfg.train.print_freq
        
        # 训练相关设置
        self.total_iters = None
        self.iter_per_epoch = None
        self.iter_count = 0
        self.imgIds = []
        
        # EMA设置
        self.ema = None
        
        # 最佳模型保存
        self.best_map = 0.0
        self.min_loss = np.inf
        
    def init_data(self):
        """初始化数据集和数据加载器"""
        from datasets.coco import COCODataSets
        
        self.train_dataset = COCODataSets(img_root=self.img_trains,
                                         annotation_path=self.train_ann,
                                         img_size=self.img_size,
                                         augments=True,
                                         aug_cfg=self.cfg.train.augmentations,
                                         use_mosaic=self.cfg.train.augmentations.mosaic,
                                         mosaic_prob=self.cfg.train.augmentations.mosaic_prob,
                                         mixup_prob=self.cfg.train.augmentations.mixup_prob)
        
        self.val_dataset = COCODataSets(img_root=self.img_vals,
                                       annotation_path=self.val_ann,
                                       img_size=self.img_size,
                                       augments=False,
                                       use_mosaic=False)
        
        self.train_dataset.set_transform(None)
        self.val_dataset.set_transform(None)
        
        # 设置批处理大小
        batch_size = self.cfg.train.batch_size
        
        # 创建数据加载器
        self.train_loader = self.train_dataset.set_attrs(
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            drop_last=True
        )
        
        self.val_loader = self.val_dataset.set_attrs(
            batch_size=self.cfg.train.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            drop_last=False
        )
        
        self.total_iters = (self.epochs - self.start_epoch) * len(self.train_loader)
        self.iter_per_epoch = len(self.train_loader)

    def init_model(self):
        """初始化模型和优化器"""
        # 导入并构建模型
        from jittor import nn
        from nets.retinanet import GFocal
        
        # 取得模型类并初始化
        self.model = GFocal(self.cfg.model)
        
        # 加载预训练权重
        if self.cfg.model.pretrained is not None and os.path.exists(self.cfg.model.pretrained):
            model_pth = jt.load(self.cfg.model.pretrained)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in model_pth['model'].items() if
                               k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Load pretrained model from {self.cfg.model.pretrained}")
        
        # 创建优化器
        self.optimizer = split_optimizer(self.model, self.cfg.train.optimizer)
        
        # 创建学习率调整器
        if self.cfg.train.lr_schedule_mode == "iter":
            self.lr_adjuster = IterWarmUpCosineDecayMultiStepLRAdjust(init_lr=self.cfg.train.optimizer.normal_lr,
                                                                     epochs=self.epochs,
                                                                     iter_per_epoch=self.iter_per_epoch,
                                                                     warmup_epoch=self.cfg.train.warmup_epoch,
                                                                     milestones=self.cfg.train.milestones,
                                                                     gamma=self.cfg.train.gamma)
        elif self.cfg.train.lr_schedule_mode == "epoch":
            self.lr_adjuster = EpochWarmUpCosineDecayLRAdjust(init_lr=self.cfg.train.optimizer.normal_lr,
                                                             epochs=self.epochs,
                                                             warmup_epoch=self.cfg.train.warmup_epoch)
        else:
            raise NotImplementedError("lr_schedule_mode only support [epoch, iter]")
        
        # 初始化EMA
        if self.use_ema:
            self.ema = ModelEMA(self.model, self.ema_decay)
        
        # 创建平均损失记录器
        if self.cfg.train.cls_loss_type == "BCE":
            self.loss_avg = {"loss": AverageLogger(), "cls_loss": AverageLogger(), "reg_loss": AverageLogger()}
            if self.cfg.model.size_dec:
                self.loss_avg.update({"size_loss": AverageLogger()})
        else:
            self.loss_avg = {"loss": AverageLogger(), "cls_loss": AverageLogger(), "reg_loss": AverageLogger(), 
                           "quality_loss": AverageLogger()}
            if self.cfg.model.size_dec:
                self.loss_avg.update({"size_loss": AverageLogger()})
        
        # 自动恢复模型
        if self.auto_resume:
            ckpt_path = os.path.join(self.save_dir, "latest.pkl")
            if os.path.exists(ckpt_path):
                self.resume(ckpt_path)

    def resume(self, ckpt_path):
        """加载检查点并恢复训练状态"""
        states = jt.load(ckpt_path)
        
        # 加载模型权重
        if isinstance(self.model, jt.nn.DataParallel):
            self.model.load_state_dict(states['model'])
        else:
            self.model.load_state_dict(states['model'])
            
        # 加载优化器状态
        self.optimizer.load_state_dict(states['optimizer'])
        
        # 加载EMA
        if self.ema is not None and states.get('ema', None) is not None:
            self.ema.ema.load_state_dict(states['ema'])
        
        # 加载其他参数
        self.start_epoch = states['epoch'] + 1
        self.best_map = states.get('best_map', 0)
        self.min_loss = states.get('min_loss', float("inf"))
        self.iter_count = states.get('iter_count', 0)
        
        # 加载混合精度训练的scaler
        if self.scaler is not None and states.get('scaler', None) is not None:
            self.scaler.load_state_dict(states['scaler'])
            
        print(f"Resume training from {ckpt_path}, start_epoch: {self.start_epoch}")

    def save(self, key_metric=None, best=False, latest=False):
        """保存模型的检查点"""
        if self.rank in [-1, 0]:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            model = self.model
            if isinstance(model, jt.nn.DataParallel):
                model = model.module
                
            ema_model = None
            if self.ema is not None:
                ema_model = self.ema.ema
            
            # 创建保存的状态字典
            states = {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_count': self.iter_count,
                'epoch': self.epoch_count,
                'best_map': self.best_map,
                'min_loss': self.min_loss
            }
            
            if self.scaler is not None:
                states['scaler'] = self.scaler.state_dict()
                
            if ema_model is not None:
                states['ema'] = ema_model.state_dict()
            
            # 保存最佳模型
            if best:
                jt.save(states, os.path.join(self.save_dir, "best.pkl"))
            
            # 保存最新模型
            if latest:
                jt.save(states, os.path.join(self.save_dir, "latest.pkl"))
            
            # 保存特定指标下的模型
            if key_metric is not None:
                jt.save(states, os.path.join(self.save_dir, f"{key_metric:.4f}.pkl"))

    def train(self, epoch):
        """单个epoch的训练步骤"""
        self.model.train()
        loss_avg = self.loss_avg
        for k in loss_avg:
            loss_avg[k].reset()
            
        # 更新学习率
        if self.cfg.train.lr_schedule_mode == "epoch":
            self.lr_adjuster.adjust_lr(self.optimizer, epoch)
            
        train_loader = self.train_loader
        start_time = time.time()
        
        for i, (imgs, targets) in enumerate(train_loader):
            # 更新学习率（按迭代次数）
            if self.cfg.train.lr_schedule_mode == "iter":
                self.lr_adjuster.adjust_lr(self.optimizer, self.iter_count)
            
            # 混合精度训练
            with jt.auto_grad():
                with jt.no_grad():
                    imgs = imgs.detach()
                
                # 前向传播，计算损失
                loss_dict = self.model(imgs, targets=targets)
                loss = loss_dict.pop('loss')
                
                # 反向传播
                self.optimizer.step(loss)
                
            # 更新EMA模型
            if self.ema is not None:
                self.ema.update(self.model)
            
            # 记录损失
            for k, v in loss_dict.items():
                loss_avg[k].update(v.item())
            loss_avg["loss"].update(loss.item())
            
            # 更新迭代计数
            self.iter_count += 1
            
            # 打印进度
            if self.iter_count % self.print_freq == 0 and self.rank in [-1, 0]:
                lr = self.optimizer.param_groups[0]['lr']
                eta_seconds = ((time.time() - start_time) / (i + 1)) * (len(train_loader) - (i + 1))
                eta_str = str(time.strftime("%H:%M:%S", time.gmtime(int(eta_seconds))))
                log_info = f"Epoch:{epoch}/{self.epochs} iter:{i}/{len(train_loader)} lr:{lr:>6f} eta:{eta_str} "
                for k, v in loss_avg.items():
                    log_info += f"{k}:{v.avg():>5f} "
                    if self.writer is not None:
                        self.writer.add_scalar(f"train/{k}", v.avg(), self.iter_count)
                if self.rank in [-1, 0]:
                    print(log_info)
                    
        # 保存模型 - 每个epoch结束时
        self.epoch_count = epoch
        self.save(latest=True)
            
    def val(self, epoch):
        """在验证集上评估模型，计算mAP等性能指标"""
        # 使用EMA模型进行评估
        if self.ema is not None:
            eval_model = self.ema.ema
        else:
            eval_model = self.model
            
        eval_model.eval()
        
        results = {}
        img_ids = []
        
        # 在验证集上进行推理
        for i, (imgs, targets) in enumerate(self.val_loader):
            # 获取图像ID
            for target in targets:
                img_ids.append(target["img_id"])
                
            # 前向传播（推理模式）
            preds = eval_model(imgs, targets=None)
            
            # 执行非极大值抑制
            preds = non_max_suppression(preds, conf_thresh=0.001, nms_thresh=0.65)
            
            # 存储结果
            for j, pred in enumerate(preds):
                if len(targets) <= j:
                    break
                img_id = targets[j]["img_id"]
                result = {"boxes": [], "scores": [], "classes": []}
                if pred is not None and len(pred) > 0:
                    boxes = pred[:, :4].float() * targets[j]["img_size"][0]
                    scores = pred[:, 4].float()
                    classes = pred[:, 5].float()
                    
                    result["boxes"] = boxes.detach().numpy()
                    result["scores"] = scores.detach().numpy()
                    result["classes"] = classes.detach().numpy()
                    
                results[img_id] = result
        
        # 准备计算mAP
        results_list = []
        targets_list = []
        for i, (_, targets) in enumerate(self.val_loader):
            for target in targets:
                img_id = target["img_id"]
                boxes = target["annos"][:, :4] * target["img_size"][0]
                cls = target["annos"][:, 4].astype(np.int32)
                targets_dict = {"boxes": boxes, "labels": cls}
                results_dict = results.get(img_id, None)
                if results_dict is None:
                    results_dict = {"boxes": np.zeros((0, 4)), "scores": np.zeros((0, )), "classes": np.zeros((0, ))}
                    
                results_list.append(results_dict)
                targets_list.append(targets_dict)
                
        # 计算mAP
        eval_result = coco_map(results_list, targets_list, img_ids=img_ids, catIds=COCO_CLASSES)
        
        # 打印结果
        map50 = eval_result["mAP"][0]
        map = eval_result["mAP"][-1]
        
        # 更新TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("val/map@0.5", map50, epoch)
            self.writer.add_scalar("val/map", map, epoch)
            if eval_result.get("AP", None) is not None:
                for i, ap in enumerate(eval_result["AP"]):
                    self.writer.add_scalar(f"val/{COCO_NAMES[i]}_ap@0.5", ap[0], epoch)
                    self.writer.add_scalar(f"val/{COCO_NAMES[i]}_ap", ap[-1], epoch)
        
        # 保存最佳模型
        if map50 > self.best_map:
            self.best_map = map50
            self.save(key_metric=map50, best=True)
            
        print(f"Epoch:{epoch}/{self.epochs} | map50: {map50:.4f} | map: {map:.4f} | best_map: {self.best_map:.4f}")
        
        # 同步最佳mAP
        self.best_map = max(map50, self.best_map)
        return map50

    def run(self):
        """运行训练"""
        # 初始化数据和模型
        self.init_data()
        self.init_model()
        
        # 如果不需要训练，直接验证
        if self.start_epoch == self.epochs:
            self.val(self.start_epoch)
            return
        
        # 训练循环
        for epoch in range(self.start_epoch, self.epochs):
            # 训练一个epoch
            self.train(epoch)
            
            # 验证
            if ((epoch + 1) % self.cfg.train.eval_epoch == 0 or epoch == self.epochs - 1) and self.no_val is False:
                self.val(epoch)

    def export_onnx(self, path):
        """导出ONNX模型"""
        self.model.eval()
        if isinstance(self.model, jt.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        # Jittor暂时不支持ONNX导出，留作后续实现
        pass 