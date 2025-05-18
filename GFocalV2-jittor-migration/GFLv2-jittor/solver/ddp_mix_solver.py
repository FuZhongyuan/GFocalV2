import os
import yaml
import jittor as jt
import numpy as np
from tqdm import tqdm
import time
import datetime
import jittor.nn as nn
from patches.torch_adapter import dist, ModelEMA, reduce_sum
from datasets.coco import COCODataSets
from nets.retinanet import GFocal
from utils.model_utils import rand_seed, AverageLogger
from metrics.map import coco_map
from utils.optims_utils import IterWarmUpCosineDecayMultiStepLRAdjust, split_optimizer

rand_seed(1024)


class DDPMixSolver(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.optim_cfg = self.cfg['optim']
        self.val_cfg = self.cfg['val']
        if self.cfg['DEBUG']:
            print("DEBUG模式已开启")
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.val_cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cfg['gpus'])
        self.gpu_num = len(self.cfg['gpus'].split(',')) if ',' in str(self.cfg['gpus']) else 1
        # Jittor不需要显式初始化分布式进程组
        
        self.tdata = COCODataSets(img_root=self.data_cfg['train_img_root'],
                                  annotation_path=self.data_cfg['train_annotation_path'],
                                  max_thresh=self.data_cfg['max_thresh'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=True,
                                  remove_blank=self.data_cfg['remove_blank']
                                  )
        self.tloader = jt.dataset.DataLoader(dataset=self.tdata,
                                          batch_size=self.data_cfg['batch_size'],
                                          num_workers=self.data_cfg['num_workers'],
                                          shuffle=True)
        self.vdata = COCODataSets(img_root=self.data_cfg['val_img_root'],
                                  annotation_path=self.data_cfg['val_annotation_path'],
                                  max_thresh=self.data_cfg['max_thresh'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )
        self.vloader = jt.dataset.DataLoader(dataset=self.vdata,
                                          batch_size=self.data_cfg['batch_size'],
                                          num_workers=self.data_cfg['num_workers'],
                                          shuffle=False)
        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata), " | ",
              "empty_data: ", self.tdata.empty_images_len)
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))
              
        # 创建模型
        self.model = GFocal(**self.model_cfg)
        self.best_map = 0.
        optimizer = split_optimizer(self.model, self.optim_cfg)
        self.local_rank = 0
        
        # Jittor不需要显式设置设备
        # 同步批归一化处理
        if self.optim_cfg['sync_bn']:
            # Jittor同步BN自动处理
            pass
            
        self.optimizer = optimizer
        self.ema = ModelEMA(self.model)
        self.lr_adjuster = IterWarmUpCosineDecayMultiStepLRAdjust(init_lr=self.optim_cfg['lr'],
                                                                  milestones=self.optim_cfg['milestones'],
                                                                  warm_up_epoch=self.optim_cfg['warm_up_epoch'],
                                                                  iter_per_epoch=len(self.tloader),
                                                                  epochs=self.optim_cfg['epochs'],
                                                                  )
        self.cls_loss_logger = AverageLogger()
        self.box_loss_logger = AverageLogger()
        self.iou_loss_logger = AverageLogger()
        self.match_num_logger = AverageLogger()
        self.loss_logger = AverageLogger()
                
        # 创建一个唯一的训练子目录
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.unique_subdir = f"{self.cfg['model_name']}_{self.model_cfg['backbone']}_{current_time}"
        self.weight_save_path = os.path.join(self.val_cfg['weight_path'], self.unique_subdir)
        os.makedirs(self.weight_save_path, exist_ok=True)
        print(f"模型权重将保存在: {self.weight_save_path}")
            

    def train(self, epoch):
        self.loss_logger.reset()
        self.cls_loss_logger.reset()
        self.box_loss_logger.reset()
        self.iou_loss_logger.reset()
        self.match_num_logger.reset()
        self.model.train()
        
        pbar = tqdm(self.tloader)
        for i, (img_tensor, targets_tensor, batch_len) in enumerate(pbar):
            _, _, h, w = img_tensor.shape
            # Jittor不需要显式移动到设备
            self.optimizer.zero_grad()
            
            # Jittor自动处理混合精度
            out = self.model(img_tensor, targets={"target": targets_tensor, "batch_len": batch_len})
            loss_qfl = out['loss_qfl']
            loss_iou = out['loss_iou']
            loss_dfl = out['loss_dfl']
            match_num = out['match_num']
            loss = loss_qfl + loss_iou + loss_dfl
            
            # 在Jittor中，只需要调用loss.sync()来进行梯度计算和反向传播
            self.optimizer.backward(loss)
            self.lr_adjuster(self.optimizer, i, epoch)
            self.optimizer.step()
            
            self.ema.update(self.model)
            lr = self.optimizer.param_groups[0]['lr']
            self.loss_logger.update(loss.item())
            self.iou_loss_logger.update(loss_iou.item())
            self.box_loss_logger.update(loss_dfl.item())
            self.cls_loss_logger.update(loss_qfl.item())
            self.match_num_logger.update(match_num)
            str_template = \
                "epoch:{:2d}|match_num:{:0>4d}|size:{:3d}|loss:{:6.4f}|qfl:{:6.4f}|dfl:{:6.4f}|iou:{:6.4f}|lr:{:8.6f}"
            pbar.set_description(str_template.format(
                epoch,
                match_num,
                h,
                self.loss_logger.avg(),
                self.cls_loss_logger.avg(),
                self.box_loss_logger.avg(),
                self.iou_loss_logger.avg(),
                lr)
            )
            
        self.ema.update_attr(self.model)
        loss_avg = reduce_sum(jt.array(self.loss_logger.avg())) / self.gpu_num
        iou_loss_avg = reduce_sum(jt.array(self.iou_loss_logger.avg())).item() / self.gpu_num
        box_loss_avg = reduce_sum(jt.array(self.box_loss_logger.avg())).item() / self.gpu_num
        cls_loss_avg = reduce_sum(jt.array(self.cls_loss_logger.avg())).item() / self.gpu_num
        match_num_sum = reduce_sum(jt.array(self.match_num_logger.sum())).item() / self.gpu_num
        
        final_template = "epoch:{:2d}|match_num:{:d}|loss:{:6.4f}|qfl:{:6.4f}|dfl:{:6.4f}|iou:{:6.4f}"
        print(final_template.format(
            epoch,
            int(match_num_sum),
            loss_avg,
            cls_loss_avg,
            box_loss_avg,
            iou_loss_avg
        ))

    def val(self, epoch):
        predict_list = list()
        target_list = list()
        self.model.eval()
        self.ema.ema.eval()
        
        pbar = tqdm(self.vloader)
        for img_tensor, targets_tensor, batch_len in pbar:
            # Jittor不需要with torch.no_grad()
            predicts = self.ema.ema(img_tensor)['predicts']
            for pred, target, info in zip(predicts, targets_tensor, batch_len):
                if pred is None:
                    continue
                predict_list.append(pred.detach().numpy())
                target_list.append(target.detach().numpy())
        mp, mr, map50, mean_ap = coco_map(predict_list, target_list)
        print("epoch: {:2d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
              .format(epoch + 1, mp, mr, map50, mean_ap))
        if mean_ap > self.best_map:
            self.save_model(epoch, mean_ap)
            self.best_map = mean_ap

    def save_model(self, epoch, map):
        jt.save(
            {
                'ema': self.ema.ema.state_dict(),
                'map': map,
                'epoch': epoch
            }, os.path.join(
                self.weight_save_path,
                '{:s}_{:s}_best.pkl'.format(self.cfg['model_name'], self.model_cfg['backbone'])
            )
        )

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
        print("训练完成，最佳mAP: {:6.4f}".format(self.best_map)) 