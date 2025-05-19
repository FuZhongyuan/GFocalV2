#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import re
import json
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("framework_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置文件路径
JITTOR_CONFIG = "/root/data-fs/GFocalV2/JittorDet/configs/gfl/gfl_r50_fpn_coco_1x.yml"
PYTORCH_CONFIG = "/root/data-fs/GFocalV2/mmdetection/configs/gfl/gfl_r50_fpn_1x_coco.py"

# 工作目录
JITTOR_WORKDIR = "/root/data-fs/GFocalV2/JittorDet"
PYTORCH_WORKDIR = "/root/data-fs/GFocalV2/mmdetection"

class LogParser:
    """解析训练日志并提取关键信息"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics = {
            'epochs': [],
            'iters': [],
            'loss': [],
            'loss_cls': [],
            'loss_bbox': [],
            'loss_dfl': [],
            'lr': [],
            'time': []
        }
        self.eval_results = {}
        
    def parse_jittor_log(self):
        """解析Jittor框架的日志文件"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取训练损失
        train_pattern = r"Epoch\s+\[(\d+)\/\d+\]\s+Iter\s+\[(\d+)\/\d+\].*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?lr:\s+([\d\.e\-]+).*?time:\s+([\d\.]+)"
        for match in re.finditer(train_pattern, content):
            epoch, iter_num, loss, loss_cls, loss_bbox, loss_dfl, lr, time_cost = match.groups()
            self.metrics['epochs'].append(int(epoch))
            self.metrics['iters'].append(int(iter_num))
            self.metrics['loss'].append(float(loss))
            self.metrics['loss_cls'].append(float(loss_cls))
            self.metrics['loss_bbox'].append(float(loss_bbox))
            self.metrics['loss_dfl'].append(float(loss_dfl))
            self.metrics['lr'].append(float(lr))
            self.metrics['time'].append(float(time_cost))
            
        # 提取评估结果
        eval_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        eval_matches = re.findall(eval_pattern, content)
        if eval_matches:
            self.eval_results['mAP'] = float(eval_matches[-1])
            
        return self.metrics, self.eval_results
    
    def parse_pytorch_log(self):
        """解析PyTorch框架的日志文件"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取训练损失
        train_pattern = r"Epoch\s+\[(\d+)\/\d+\]\s+Iter\s+\[(\d+)\/\d+\].*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?lr:\s+([\d\.e\-]+).*?time:\s+([\d\.]+)"
        for match in re.finditer(train_pattern, content):
            epoch, iter_num, loss, loss_cls, loss_bbox, loss_dfl, lr, time_cost = match.groups()
            self.metrics['epochs'].append(int(epoch))
            self.metrics['iters'].append(int(iter_num))
            self.metrics['loss'].append(float(loss))
            self.metrics['loss_cls'].append(float(loss_cls))
            self.metrics['loss_bbox'].append(float(loss_bbox))
            self.metrics['loss_dfl'].append(float(loss_dfl))
            self.metrics['lr'].append(float(lr))
            self.metrics['time'].append(float(time_cost))
            
        # 提取评估结果 (查找最近的coco格式结果)
        eval_pattern = r"bbox_mAP: ([\d\.]+)"
        eval_matches = re.findall(eval_pattern, content)
        if eval_matches:
            self.eval_results['mAP'] = float(eval_matches[-1])
            
        return self.metrics, self.eval_results


def run_training(framework, max_epochs=None, max_iters=None):
    """运行指定框架的训练过程"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if framework == 'jittor':
        config = JITTOR_CONFIG
        work_dir = f"work_dirs/gfl_jittor_{timestamp}"
        cmd = f"cd {JITTOR_WORKDIR} && python tools/train.py {config} --work-dir {work_dir}"
        log_file = os.path.join(JITTOR_WORKDIR, work_dir, "train.log")
    else:  # pytorch
        config = PYTORCH_CONFIG
        work_dir = f"work_dirs/gfl_pytorch_{timestamp}"
        cmd = f"cd {PYTORCH_WORKDIR} && python tools/train.py {config} --work-dir {work_dir}"
        log_file = os.path.join(PYTORCH_WORKDIR, work_dir, "train.log")
    
    if max_epochs:
        cmd += f" --cfg-options runner.max_epochs={max_epochs}"
    if max_iters:
        cmd += f" --cfg-options runner.max_iters={max_iters}"
    
    # 创建工作目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.info(f"开始在 {framework} 框架下运行训练...")
    logger.info(f"运行命令: {cmd}")
    logger.info(f"日志将保存到: {log_file}")
    
    # 使用subprocess运行命令，并将输出写入日志文件
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时读取输出并写入日志
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        
        process.wait()
    
    return_code = process.returncode
    logger.info(f"{framework} 训练完成，返回码: {return_code}")
    
    return log_file

def generate_comparison_plots(jittor_metrics, pytorch_metrics, output_dir):
    """生成两个框架的对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建一个包含多个子图的大图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 损失曲线对比
    axes[0, 0].plot(jittor_metrics['iters'], jittor_metrics['loss'], label='Jittor')
    axes[0, 0].plot(pytorch_metrics['iters'], pytorch_metrics['loss'], label='PyTorch')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 分类损失对比
    axes[0, 1].plot(jittor_metrics['iters'], jittor_metrics['loss_cls'], label='Jittor')
    axes[0, 1].plot(pytorch_metrics['iters'], pytorch_metrics['loss_cls'], label='PyTorch')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Iterations')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 边界框损失对比
    axes[1, 0].plot(jittor_metrics['iters'], jittor_metrics['loss_bbox'], label='Jittor')
    axes[1, 0].plot(pytorch_metrics['iters'], pytorch_metrics['loss_bbox'], label='PyTorch')
    axes[1, 0].set_title('Bounding Box Loss')
    axes[1, 0].set_xlabel('Iterations')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # DFL损失对比
    axes[1, 1].plot(jittor_metrics['iters'], jittor_metrics['loss_dfl'], label='Jittor')
    axes[1, 1].plot(pytorch_metrics['iters'], pytorch_metrics['loss_dfl'], label='PyTorch')
    axes[1, 1].set_title('Distribution Focal Loss')
    axes[1, 1].set_xlabel('Iterations')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    logger.info(f"损失对比图已保存到 {os.path.join(output_dir, 'loss_comparison.png')}")
    
    # 训练时间对比
    plt.figure(figsize=(10, 6))
    plt.plot(jittor_metrics['iters'], jittor_metrics['time'], label='Jittor')
    plt.plot(pytorch_metrics['iters'], pytorch_metrics['time'], label='PyTorch')
    plt.title('Training Time per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    logger.info(f"时间对比图已保存到 {os.path.join(output_dir, 'time_comparison.png')}")

def generate_report(jittor_metrics, pytorch_metrics, jittor_eval, pytorch_eval, output_dir):
    """生成框架对比报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'framework_comparison_report.md')
    
    # 计算均值和标准差
    jittor_avg_loss = np.mean(jittor_metrics['loss'])
    pytorch_avg_loss = np.mean(pytorch_metrics['loss'])
    
    jittor_avg_time = np.mean(jittor_metrics['time'])
    pytorch_avg_time = np.mean(pytorch_metrics['time'])
    
    # 创建报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# GFL在Jittor和PyTorch框架下的对比报告\n\n")
        f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 训练配置\n\n")
        f.write(f"Jittor配置文件: `{JITTOR_CONFIG}`\n\n")
        f.write(f"PyTorch配置文件: `{PYTORCH_CONFIG}`\n\n")
        
        f.write("## 损失函数对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 差异比例 |\n")
        f.write("|------|--------|---------|----------|\n")
        
        diff_pct = abs(jittor_avg_loss - pytorch_avg_loss) / pytorch_avg_loss * 100
        f.write(f"| 平均总损失 | {jittor_avg_loss:.4f} | {pytorch_avg_loss:.4f} | {diff_pct:.2f}% |\n")
        
        jittor_avg_cls = np.mean(jittor_metrics['loss_cls'])
        pytorch_avg_cls = np.mean(pytorch_metrics['loss_cls'])
        diff_pct = abs(jittor_avg_cls - pytorch_avg_cls) / pytorch_avg_cls * 100
        f.write(f"| 平均分类损失 | {jittor_avg_cls:.4f} | {pytorch_avg_cls:.4f} | {diff_pct:.2f}% |\n")
        
        jittor_avg_bbox = np.mean(jittor_metrics['loss_bbox'])
        pytorch_avg_bbox = np.mean(pytorch_metrics['loss_bbox'])
        diff_pct = abs(jittor_avg_bbox - pytorch_avg_bbox) / pytorch_avg_bbox * 100
        f.write(f"| 平均边界框损失 | {jittor_avg_bbox:.4f} | {pytorch_avg_bbox:.4f} | {diff_pct:.2f}% |\n")
        
        jittor_avg_dfl = np.mean(jittor_metrics['loss_dfl'])
        pytorch_avg_dfl = np.mean(pytorch_metrics['loss_dfl'])
        diff_pct = abs(jittor_avg_dfl - pytorch_avg_dfl) / pytorch_avg_dfl * 100
        f.write(f"| 平均DFL损失 | {jittor_avg_dfl:.4f} | {pytorch_avg_dfl:.4f} | {diff_pct:.2f}% |\n\n")
        
        f.write("## 训练性能对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 比例 |\n")
        f.write("|------|--------|---------|------|\n")
        
        speedup = pytorch_avg_time / jittor_avg_time
        f.write(f"| 平均每次迭代时间 | {jittor_avg_time:.4f}秒 | {pytorch_avg_time:.4f}秒 | {speedup:.2f}x |\n")
        
        if jittor_eval and pytorch_eval and 'mAP' in jittor_eval and 'mAP' in pytorch_eval:
            f.write("\n## 评估结果对比\n\n")
            f.write("| 指标 | Jittor | PyTorch | 差异比例 |\n")
            f.write("|------|--------|---------|----------|\n")
            
            diff_pct = abs(jittor_eval['mAP'] - pytorch_eval['mAP']) / pytorch_eval['mAP'] * 100
            f.write(f"| mAP | {jittor_eval['mAP']:.4f} | {pytorch_eval['mAP']:.4f} | {diff_pct:.2f}% |\n")
        
        f.write("\n## 损失曲线\n\n")
        f.write("![Loss Comparison](loss_comparison.png)\n\n")
        
        f.write("## 训练时间\n\n")
        f.write("![Time Comparison](time_comparison.png)\n\n")
        
        f.write("## 结论\n\n")
        f.write("1. **损失函数对齐情况**: ")
        if diff_pct < 5:
            f.write("两个框架的损失函数对齐良好，差异小于5%。\n")
        else:
            f.write(f"两个框架的损失函数存在一定差异，差异约为{diff_pct:.2f}%。\n")
            
        f.write("2. **训练性能**: ")
        if speedup > 1:
            f.write(f"Jittor框架的训练速度比PyTorch快{speedup:.2f}倍。\n")
        else:
            f.write(f"PyTorch框架的训练速度比Jittor快{1/speedup:.2f}倍。\n")
            
        if jittor_eval and pytorch_eval and 'mAP' in jittor_eval and 'mAP' in pytorch_eval:
            f.write("3. **评估结果**: ")
            if diff_pct < 5:
                f.write("两个框架的评估结果对齐良好，mAP差异小于5%。\n")
            else:
                f.write(f"两个框架的评估结果存在一定差异，mAP差异约为{diff_pct:.2f}%。\n")
    
    logger.info(f"对比报告已保存到 {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description='对比PyTorch和Jittor框架下的GFL实现')
    parser.add_argument('--epochs', type=int, default=1, help='训练的最大轮数')
    parser.add_argument('--iters', type=int, help='训练的最大迭代数')
    parser.add_argument('--output-dir', type=str, default='comparison_results', help='结果输出目录')
    parser.add_argument('--jittor-only', action='store_true', help='仅运行Jittor框架')
    parser.add_argument('--pytorch-only', action='store_true', help='仅运行PyTorch框架')
    parser.add_argument('--jittor-log', type=str, help='指定Jittor日志文件(不运行训练)')
    parser.add_argument('--pytorch-log', type=str, help='指定PyTorch日志文件(不运行训练)')
    
    args = parser.parse_args()
    
    logger.info("开始GFL框架对比测试")
    logger.info(f"参数: {args}")
    
    jittor_log_file = None
    pytorch_log_file = None
    
    # 运行训练或加载指定的日志文件
    if args.jittor_log:
        jittor_log_file = args.jittor_log
        logger.info(f"使用指定的Jittor日志文件: {jittor_log_file}")
    elif not args.pytorch_only:
        jittor_log_file = run_training('jittor', args.epochs, args.iters)
    
    if args.pytorch_log:
        pytorch_log_file = args.pytorch_log
        logger.info(f"使用指定的PyTorch日志文件: {pytorch_log_file}")
    elif not args.jittor_only:
        pytorch_log_file = run_training('pytorch', args.epochs, args.iters)
    
    # 解析日志
    if jittor_log_file:
        jittor_parser = LogParser(jittor_log_file)
        jittor_metrics, jittor_eval = jittor_parser.parse_jittor_log()
        logger.info(f"成功解析Jittor日志，提取了{len(jittor_metrics['loss'])}条记录")
    else:
        jittor_metrics, jittor_eval = None, None
    
    if pytorch_log_file:
        pytorch_parser = LogParser(pytorch_log_file)
        pytorch_metrics, pytorch_eval = pytorch_parser.parse_pytorch_log()
        logger.info(f"成功解析PyTorch日志，提取了{len(pytorch_metrics['loss'])}条记录")
    else:
        pytorch_metrics, pytorch_eval = None, None
    
    # 生成对比报告
    if jittor_metrics and pytorch_metrics:
        output_dir = os.path.abspath(args.output_dir)
        generate_comparison_plots(jittor_metrics, pytorch_metrics, output_dir)
        report_file = generate_report(jittor_metrics, pytorch_metrics, jittor_eval, pytorch_eval, output_dir)
        logger.info(f"对比完成! 报告保存在: {report_file}")
    elif jittor_metrics:
        logger.info("只有Jittor框架的结果，无法生成对比报告")
    elif pytorch_metrics:
        logger.info("只有PyTorch框架的结果，无法生成对比报告")
    else:
        logger.error("没有任何框架的结果，无法生成报告")

if __name__ == '__main__':
    main() 