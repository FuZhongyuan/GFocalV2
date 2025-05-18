#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import jittor as jt
import argparse
import sys
import yaml
import logging
from argparse import Namespace
from easydict import EasyDict
from solver.ddp_mix_solver import DDPMixSolver
from utils.model_utils import rand_seed


def get_args():
    parser = argparse.ArgumentParser(description="Train GFocalV2 on COCO")
    parser.add_argument("--config-file", default="config/gfocal.yaml", type=str, help="config file")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("--work-dir", default="work_dir", type=str, help="work directory")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_yaml(cfg, args):
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    cfg.update(config)  # 更新配置信息
    # 处理命令行参数覆盖配置文件
    if args.opts:
        opts_dict = {}
        for i in range(0, len(args.opts), 2):
            key = args.opts[i].replace("--", "")
            value = args.opts[i + 1]
            try:
                value = eval(value)
            except:
                pass
            opts_dict[key] = value
        cfg.update(opts_dict)


def main():
    args = get_args()
    cfg = EasyDict()
    load_yaml(cfg, args)
    
    # 设置随机种子
    rand_seed(args.seed)
    cfg.seed = args.seed
    
    # 设置模型恢复
    cfg.model.resume = args.resume
    
    # 创建工作目录
    os.makedirs(args.work_dir, exist_ok=True)
    
    # 设置 Jittor 环境
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    
    # 是否使用分布式训练
    cfg.rank = -1
    cfg.local_rank = args.local_rank
    
    # 创建 solver 并启动训练
    solver = DDPMixSolver(cfg)
    solver.run()


if __name__ == "__main__":
    main() 