#!/usr/bin/env python
# 使用 GFocalV2 进行目标检测的示例脚本

"""
这是一个如何使用精简后的 GFocalV2-jittor-migration 代码库进行目标检测的示例脚本。
"""

import os
import os.path as osp
import argparse

from jittordet.engine import Runner, load_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='GFocalV2 目标检测示例')
    parser.add_argument('--config', default='configs/gfl/gfl_r50_fpn_coco_1x.yml', help='配置文件路径')
    parser.add_argument('--work-dir', default='./work_dirs/gfl_r50_fpn_coco_1x', help='保存日志和模型的目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--disable-cuda', action='store_true', help='禁用 CUDA 并使用 CPU 训练')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='训练或测试模式')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # 设置随机种子
    if args.seed is not None:
        cfg.seed = args.seed
    
    # 设置是否禁用 CUDA
    cfg.disable_cuda = args.disable_cuda

    runner = Runner.from_cfg(cfg)
    
    if args.mode == 'train':
        runner.train()
    else:
        runner.test()


if __name__ == '__main__':
    main() 