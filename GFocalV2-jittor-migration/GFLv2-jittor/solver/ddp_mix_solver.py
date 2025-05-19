import os
import yaml
import jittor as jt
import numpy as np
from tqdm import tqdm
import time
import datetime
import jittor.nn as nn
# from jittor import amp
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
              
        model = GFocal(**self.model_cfg)
        self.best_map = 0.
        optimizer = split_optimizer(model, self.optim_cfg)
        local_rank = dist.get_rank()
        self.local_rank = local_rank
        # if self.optim_cfg['sync_bn']:
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # self.model = nn.parallel.distributed.DistributedDataParallel(model,
        #                                                              device_ids=[local_rank],
        #                                                              output_device=local_rank)
        self.model = model
        self.scaler = None
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
        if dist.get_rank() == 0:
            os.makedirs(self.weight_save_path, exist_ok=True)
            print(f"模型权重将保存在: {self.weight_save_path}")
            

    def train(self, epoch):
        self.loss_logger.reset()
        self.cls_loss_logger.reset()
        self.box_loss_logger.reset()
        self.iou_loss_logger.reset()
        self.match_num_logger.reset()
        self.model.train()
        if self.local_rank == 0:
            pbar = tqdm(self.tloader)
        else:
            pbar = self.tloader
        for i, (img_tensor, targets_tensor, batch_len) in enumerate(pbar):
            _, _, h, w = img_tensor.shape
            self.optimizer.zero_grad()
            if self.scaler is not None:
                # with jt.amp.autocast(device_type='cuda',enabled=True):
                out = self.model(img_tensor,
                                    targets={"target": targets_tensor, "batch_len": batch_len})
                loss_qfl = out['loss_qfl']
                loss_iou = out['loss_iou']
                loss_dfl = out['loss_dfl']
                match_num = out['match_num']
                loss = loss_qfl + loss_iou + loss_dfl
                self.scaler.scale(loss).backward()
                self.lr_adjuster(self.optimizer, i, epoch)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(img_tensor,
                                 targets={"target": targets_tensor, "batch_len": batch_len})
                loss_qfl = out['loss_qfl']
                loss_iou = out['loss_iou']
                loss_dfl = out['loss_dfl']
                match_num = out['match_num']
                loss = loss_qfl + loss_iou + loss_dfl
                loss.backward()
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
            if self.local_rank == 0:
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
        loss_avg = reduce_sum(jt.Var(self.loss_logger.avg())) / self.gpu_num
        iou_loss_avg = reduce_sum(jt.Var(self.iou_loss_logger.avg())).item() / self.gpu_num
        box_loss_avg = reduce_sum(jt.Var(self.box_loss_logger.avg())).item() / self.gpu_num
        cls_loss_avg = reduce_sum(jt.Var(self.cls_loss_logger.avg())).item() / self.gpu_num
        match_num_sum = reduce_sum(jt.Var(self.match_num_logger.sum())).item() / self.gpu_num
        if self.local_rank == 0:
            final_template = "epoch:{:2d}|match_num:{:d}|loss:{:6.4f}|qfl:{:6.4f}|dfl:{:6.4f}|iou:{:6.4f}"
            print(final_template.format(
                epoch,
                int(match_num_sum),
                loss_avg,
                cls_loss_avg,
                box_loss_avg,
                iou_loss_avg
            ))

    @jt.no_grad()
    def val(self, epoch):
        predict_list = list()
        target_list = list()
        self.model.eval()
        self.ema.ema.eval()
        if self.local_rank == 0:
            pbar = tqdm(self.vloader)
        else:
            pbar = self.vloader
        for img_tensor, targets_tensor, batch_len in pbar:
            predicts = self.ema.ema(img_tensor)['predicts']
            for pred, target in zip(predicts, targets_tensor.split(batch_len)):
                predict_list.append(pred)
                target_list.append(target)
        mp, mr, map50, mean_ap = coco_map(predict_list, target_list)
        mp = reduce_sum(jt.Var(mp)) / self.gpu_num
        mr = reduce_sum(jt.Var(mr)) / self.gpu_num
        map50 = reduce_sum(jt.Var(map50)) / self.gpu_num
        mean_ap = reduce_sum(jt.Var(mean_ap)) / self.gpu_num

        if self.local_rank == 0:
            print("*" * 20, "eval start", "*" * 20)
            print("epoch: {:2d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
                  .format(epoch + 1,
                          mp * 100,
                          mr * 100,
                          map50 * 100,
                          mean_ap * 100))
            print("*" * 20, "eval end", "*" * 20)
        
        # 使用唯一子目录保存模型权重
        last_weight_path = os.path.join(self.weight_save_path,
                                        "{:s}_{:s}_epoch{:03d}_last.pth"
                                        .format(self.cfg['model_name'],
                                                self.model_cfg['backbone'],
                                                epoch + 1))
        best_map_weight_path = os.path.join(self.weight_save_path,
                                            "{:s}_{:s}_best_map.pth"
                                            .format(self.cfg['model_name'],
                                                    self.model_cfg['backbone']))
        ema_static = self.ema.ema.state_dict()
        cpkt = {
            "ema": ema_static,
            "map": mean_ap * 100,
            "epoch": epoch,
        }
        if self.local_rank != 0:
            return
            
        print(f"保存权重到 {last_weight_path}")
        jt.save(cpkt, last_weight_path)
        if mean_ap > self.best_map:
            jt.save(cpkt, best_map_weight_path)
            self.best_map = mean_ap

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
