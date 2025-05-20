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
JITTOR_CONFIG = "/root/data-fs/GFocalV2/GFocalV2Jittor/configs/gfl/gfl_r50_fpn_coco_1x_enhanced.yml"
PYTORCH_CONFIG = "/root/data-fs/GFocalV2/GFocalV2Pytorch/configs/gfl/gfl_r50_fpn_1x_coco.py"

# 工作目录
JITTOR_WORKDIR = "/root/data-fs/GFocalV2/GFocalV2Jittor"
PYTORCH_WORKDIR = "/root/data-fs/GFocalV2/GFocalV2Pytorch"

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
        self.eval_results = {
            'mAP': None,
            'mAP_50': None,
            'mAP_75': None,
            'mAP_s': None,
            'mAP_m': None,
            'mAP_l': None,
        }
        
    def parse_jittor_log(self):
        """解析Jittor框架的日志文件"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查是否有训练损失记录
        train_records = []
        
        # 尝试解析Epoch级别的训练信息
        train_pattern = r"Epoch\s+\[(\d+)/\d+\]\[(\d+)/\d+\].*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?lr:\s+([\d\.e\-]+).*?time:\s+([\d\.]+)"
        for match in re.finditer(train_pattern, content):
            epoch, iter_num, loss, loss_cls, loss_bbox, loss_dfl, lr, time_cost = match.groups()
            train_records.append((int(epoch), int(iter_num), float(loss), float(loss_cls), float(loss_bbox), float(loss_dfl), float(lr), float(time_cost)))
        
        # 如果没有找到Epoch级别的记录，尝试解析Iter级别的记录
        if not train_records:
            iter_pattern = r"Iter\s+\[(\d+)/\d+\].*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?lr:\s+([\d\.e\-]+).*?time:\s+([\d\.]+)"
            for match in re.finditer(iter_pattern, content):
                iter_num, loss, loss_cls, loss_bbox, loss_dfl, lr, time_cost = match.groups()
                # 根据迭代次数估算epoch
                epoch = int(int(iter_num) / 100) + 1  # 假设每个epoch有100次迭代
                train_records.append((epoch, int(iter_num), float(loss), float(loss_cls), float(loss_bbox), float(loss_dfl), float(lr), float(time_cost)))
        
        # 如果还是没有找到记录，尝试解析简化的日志格式
        if not train_records:
            simple_pattern = r"Epoch (\d+).*?iter (\d+).*?loss=([\d\.]+)"
            for match in re.finditer(simple_pattern, content):
                epoch, iter_num, loss = match.groups()
                # 由于缺少详细的损失分解，使用统一的值
                train_records.append((int(epoch), int(iter_num), float(loss), 0.0, 0.0, 0.0, 0.0, 0.0))
        
        # 将解析结果添加到metrics中
        for record in train_records:
            epoch, iter_num, loss, loss_cls, loss_bbox, loss_dfl, lr, time_cost = record
            self.metrics['epochs'].append(epoch)
            self.metrics['iters'].append(iter_num)
            self.metrics['loss'].append(loss)
            self.metrics['loss_cls'].append(loss_cls)
            self.metrics['loss_bbox'].append(loss_bbox)
            self.metrics['loss_dfl'].append(loss_dfl)
            self.metrics['lr'].append(lr)
            self.metrics['time'].append(time_cost)
            
        # 检查是否缺少loss记录
        if not self.metrics['loss']:
            logger.warning("Jittor日志中没有找到loss记录，可能需要修改训练代码添加loss打印")
        
        # 提取评估结果 (mAP等)
        empty_dataset_error = "The testing results of the whole dataset is empty" in content
        if empty_dataset_error:
            logger.warning("检测到测试结果为空的错误，这可能需要重新运行测试")
        
        # 尝试解析mAP结果 (coco格式)
        bbox_map_pattern = r"bbox_mAP: ([\d\.]+)\s+bbox_mAP_50: ([\d\.]+)\s+bbox_mAP_75: ([\d\.]+)\s+bbox_mAP_s: ([\d\.]+)\s+bbox_mAP_m: ([\d\.]+)\s+bbox_mAP_l: ([\d\.]+)"
        bbox_map_matches = re.findall(bbox_map_pattern, content)
        
        if bbox_map_matches:
            # 使用最后一次评估结果
            last_match = bbox_map_matches[-1]
            self.eval_results['mAP'] = float(last_match[0])
            self.eval_results['mAP_50'] = float(last_match[1])
            self.eval_results['mAP_75'] = float(last_match[2])
            self.eval_results['mAP_s'] = float(last_match[3])
            self.eval_results['mAP_m'] = float(last_match[4])
            self.eval_results['mAP_l'] = float(last_match[5])
        else:
            # 尝试找到原始COCO评估格式的结果
            ap_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
            ap_50_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50\s+\|\s+area=\s+all\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_75_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.75\s+\|\s+area=\s+all\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_s_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s*small\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_m_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s*medium\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_l_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s*large\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            
            # 查找所有匹配项并取最后一个
            ap_matches = re.findall(ap_pattern, content)
            ap_50_matches = re.findall(ap_50_pattern, content)
            ap_75_matches = re.findall(ap_75_pattern, content)
            ap_s_matches = re.findall(ap_s_pattern, content)
            ap_m_matches = re.findall(ap_m_pattern, content)
            ap_l_matches = re.findall(ap_l_pattern, content)
            
            if ap_matches:
                self.eval_results['mAP'] = float(ap_matches[-1])
            if ap_50_matches:
                self.eval_results['mAP_50'] = float(ap_50_matches[-1])
            if ap_75_matches:
                self.eval_results['mAP_75'] = float(ap_75_matches[-1])
            if ap_s_matches:
                self.eval_results['mAP_s'] = float(ap_s_matches[-1])
            if ap_m_matches:
                self.eval_results['mAP_m'] = float(ap_m_matches[-1])
            if ap_l_matches:
                self.eval_results['mAP_l'] = float(ap_l_matches[-1])
        
        return self.metrics, self.eval_results
    
    def parse_pytorch_log(self):
        """解析PyTorch框架的日志文件"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取训练损失
        train_pattern = r"Epoch\(train\)\s+\[(\d+)\]\[(\d+)/\d+\].*?lr:\s+([\d\.e\-]+).*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?time:\s+([\d\.]+)"
        for match in re.finditer(train_pattern, content):
            epoch, iter_num, lr, loss, loss_cls, loss_bbox, loss_dfl, time_cost = match.groups()
            self.metrics['epochs'].append(int(epoch))
            self.metrics['iters'].append(int(iter_num))
            self.metrics['loss'].append(float(loss))
            self.metrics['loss_cls'].append(float(loss_cls))
            self.metrics['loss_bbox'].append(float(loss_bbox))
            self.metrics['loss_dfl'].append(float(loss_dfl))
            self.metrics['lr'].append(float(lr))
            self.metrics['time'].append(float(time_cost))
        
        # 如果没有找到记录，尝试不同的格式
        if not self.metrics['loss']:
            alt_pattern = r"Epoch\s+\[(\d+)/\d+\]\s+Iter\s+\[(\d+)/\d+\].*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?lr:\s+([\d\.e\-]+).*?time:\s+([\d\.]+)"
            for match in re.finditer(alt_pattern, content):
                epoch, iter_num, loss, loss_cls, loss_bbox, loss_dfl, lr, time_cost = match.groups()
                self.metrics['epochs'].append(int(epoch))
                self.metrics['iters'].append(int(iter_num))
                self.metrics['loss'].append(float(loss))
                self.metrics['loss_cls'].append(float(loss_cls))
                self.metrics['loss_bbox'].append(float(loss_bbox))
                self.metrics['loss_dfl'].append(float(loss_dfl))
                self.metrics['lr'].append(float(lr))
                self.metrics['time'].append(float(time_cost))
        
        # 检查是否缺少loss记录
        if not self.metrics['loss']:
            logger.warning("PyTorch日志中没有找到loss记录，可能需要修改训练代码添加loss打印")
            
        # 提取评估结果
        bbox_map_pattern = r"coco/bbox_mAP:\s+([\d\.]+)\s+coco/bbox_mAP_50:\s+([\d\.]+)\s+coco/bbox_mAP_75:\s+([\d\.]+)\s+coco/bbox_mAP_s:\s+([\d\.]+)\s+coco/bbox_mAP_m:\s+([\d\.]+)\s+coco/bbox_mAP_l:\s+([\d\.]+)"
        bbox_map_matches = re.findall(bbox_map_pattern, content)
        
        if bbox_map_matches:
            # 使用最后一次评估结果
            last_match = bbox_map_matches[-1]
            self.eval_results['mAP'] = float(last_match[0])
            self.eval_results['mAP_50'] = float(last_match[1])
            self.eval_results['mAP_75'] = float(last_match[2])
            self.eval_results['mAP_s'] = float(last_match[3])
            self.eval_results['mAP_m'] = float(last_match[4])
            self.eval_results['mAP_l'] = float(last_match[5])
        else:
            # 尝试找到原始COCO评估格式的结果
            ap_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
            ap_50_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50\s+\|\s+area=\s+all\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_75_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.75\s+\|\s+area=\s+all\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_s_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s*small\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_m_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s*medium\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            ap_l_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s*large\s+\|\s+maxDets=\d+\s+\]\s+=\s+([\d\.]+)"
            
            # 查找所有匹配项并取最后一个
            ap_matches = re.findall(ap_pattern, content)
            ap_50_matches = re.findall(ap_50_pattern, content)
            ap_75_matches = re.findall(ap_75_pattern, content)
            ap_s_matches = re.findall(ap_s_pattern, content)
            ap_m_matches = re.findall(ap_m_pattern, content)
            ap_l_matches = re.findall(ap_l_pattern, content)
            
            if ap_matches:
                self.eval_results['mAP'] = float(ap_matches[-1])
            if ap_50_matches:
                self.eval_results['mAP_50'] = float(ap_50_matches[-1])
            if ap_75_matches:
                self.eval_results['mAP_75'] = float(ap_75_matches[-1])
            if ap_s_matches:
                self.eval_results['mAP_s'] = float(ap_s_matches[-1])
            if ap_m_matches:
                self.eval_results['mAP_m'] = float(ap_m_matches[-1])
            if ap_l_matches:
                self.eval_results['mAP_l'] = float(ap_l_matches[-1])
            
        return self.metrics, self.eval_results


def run_training(framework, max_epochs=None, max_iters=None):
    """运行指定框架的训练过程"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if framework == 'jittor':
        config = JITTOR_CONFIG
        work_dir = f"work_dirs/gfl_jittor_{timestamp}"
        cmd = f"cd {JITTOR_WORKDIR} && pip install -e . && python tools/train.py {config} --work-dir {work_dir}"
        log_file = os.path.join(JITTOR_WORKDIR, work_dir, "train.log")
    else:  # pytorch
        config = PYTORCH_CONFIG
        work_dir = f"work_dirs/gfl_pytorch_{timestamp}"
        cmd = f"cd {PYTORCH_WORKDIR} && pip install -e . && python tools/train.py {config} --work-dir {work_dir}"
        log_file = os.path.join(PYTORCH_WORKDIR, work_dir, "train.log")
    
    # if max_epochs:
    #     cmd += f" --cfg-options runner.max_epochs={max_epochs}"
    # if max_iters:
    #     cmd += f" --cfg-options runner.max_iters={max_iters}"
    
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

def generate_map_comparison_plot(jittor_eval, pytorch_eval, output_dir):
    """生成mAP比较图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备绘图数据
    metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    labels = ['mAP', 'mAP@0.5', 'mAP@0.75', 'mAP (small)', 'mAP (medium)', 'mAP (large)']
    
    # 只选择有效的指标（两个框架都有值的指标）
    valid_indices = []
    for i, metric in enumerate(metrics):
        if metric in jittor_eval and metric in pytorch_eval and jittor_eval[metric] is not None and pytorch_eval[metric] is not None:
            valid_indices.append(i)
    
    if not valid_indices:
        logger.warning("没有找到有效的mAP指标，无法生成比较图表")
        return
    
    valid_metrics = [metrics[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    jittor_values = [jittor_eval[m] for m in valid_metrics]
    pytorch_values = [pytorch_eval[m] for m in valid_metrics]
    
    # 创建柱状图
    x = np.arange(len(valid_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    jittor_bars = ax.bar(x - width/2, jittor_values, width, label='Jittor', color='skyblue')
    pytorch_bars = ax.bar(x + width/2, pytorch_values, width, label='PyTorch', color='salmon')
    
    # 添加图表标签和标题
    ax.set_title('mAP Comparison between Jittor and PyTorch', fontsize=16)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=12)
    ax.legend(fontsize=12)
    
    # 添加数据标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
    
    add_labels(jittor_bars)
    add_labels(pytorch_bars)
    
    # 添加网格线和调整布局
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'map_comparison.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"mAP比较图表已保存到 {output_path}")
    
    # 如果有足够的数据点，创建雷达图
    if len(valid_metrics) >= 3:
        create_radar_chart(valid_labels, jittor_values, pytorch_values, output_dir)

def create_radar_chart(categories, jittor_values, pytorch_values, output_dir):
    """创建雷达图比较两个框架的mAP指标"""
    # 计算角度
    n = len(categories)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    
    # 闭合雷达图
    jittor_values += jittor_values[:1]
    pytorch_values += pytorch_values[:1]
    angles += angles[:1]
    categories += categories[:1]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 绘制Jittor数据
    ax.plot(angles, jittor_values, 'o-', linewidth=2, label='Jittor', color='skyblue')
    ax.fill(angles, jittor_values, alpha=0.25, color='skyblue')
    
    # 绘制PyTorch数据
    ax.plot(angles, pytorch_values, 'o-', linewidth=2, label='PyTorch', color='salmon')
    ax.fill(angles, pytorch_values, alpha=0.25, color='salmon')
    
    # 设置标签
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    
    # 添加标题和图例
    ax.set_title('mAP Metrics Comparison (Radar Chart)', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 保存图表
    output_path = os.path.join(output_dir, 'map_radar_chart.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"mAP雷达图已保存到 {output_path}")

def generate_report(jittor_metrics, pytorch_metrics, jittor_eval, pytorch_eval, output_dir):
    """生成框架对比报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'framework_comparison_report.md')
    
    # 计算均值和标准差
    jittor_avg_loss = np.mean(jittor_metrics['loss']) if jittor_metrics['loss'] else None
    pytorch_avg_loss = np.mean(pytorch_metrics['loss']) if pytorch_metrics['loss'] else None
    
    jittor_avg_time = np.mean(jittor_metrics['time']) if jittor_metrics['time'] else None
    pytorch_avg_time = np.mean(pytorch_metrics['time']) if pytorch_metrics['time'] else None
    
    # 创建报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# GFL在Jittor和PyTorch框架下的对比报告\n\n")
        f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 训练配置\n\n")
        f.write(f"Jittor配置文件: `{JITTOR_CONFIG}`\n\n")
        f.write(f"PyTorch配置文件: `{PYTORCH_CONFIG}`\n\n")
        
        # 训练记录统计
        f.write("## 训练记录统计\n\n")
        f.write("| 框架 | 记录的迭代次数 | 记录的Epoch数 |\n")
        f.write("|------|--------------|------------|\n")
        jittor_iters = len(jittor_metrics['iters']) if jittor_metrics['iters'] else 0
        jittor_epochs = len(set(jittor_metrics['epochs'])) if jittor_metrics['epochs'] else 0
        pytorch_iters = len(pytorch_metrics['iters']) if pytorch_metrics['iters'] else 0
        pytorch_epochs = len(set(pytorch_metrics['epochs'])) if pytorch_metrics['epochs'] else 0
        f.write(f"| Jittor | {jittor_iters} | {jittor_epochs} |\n")
        f.write(f"| PyTorch | {pytorch_iters} | {pytorch_epochs} |\n\n")
        
        # 损失函数对比
        f.write("## 损失函数对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 差异比例 |\n")
        f.write("|------|--------|---------|----------|\n")
        
        # 检查是否有损失数据
        if jittor_avg_loss is not None and pytorch_avg_loss is not None:
            diff_pct = abs(jittor_avg_loss - pytorch_avg_loss) / pytorch_avg_loss * 100
            f.write(f"| 平均总损失 | {jittor_avg_loss:.4f} | {pytorch_avg_loss:.4f} | {diff_pct:.2f}% |\n")
            
            if jittor_metrics['loss_cls'] and pytorch_metrics['loss_cls']:
                jittor_avg_cls = np.mean(jittor_metrics['loss_cls'])
                pytorch_avg_cls = np.mean(pytorch_metrics['loss_cls'])
                diff_pct = abs(jittor_avg_cls - pytorch_avg_cls) / pytorch_avg_cls * 100
                f.write(f"| 平均分类损失 | {jittor_avg_cls:.4f} | {pytorch_avg_cls:.4f} | {diff_pct:.2f}% |\n")
            else:
                f.write("| 平均分类损失 | 数据不可用 | 数据不可用 | - |\n")
            
            if jittor_metrics['loss_bbox'] and pytorch_metrics['loss_bbox']:
                jittor_avg_bbox = np.mean(jittor_metrics['loss_bbox'])
                pytorch_avg_bbox = np.mean(pytorch_metrics['loss_bbox'])
                diff_pct = abs(jittor_avg_bbox - pytorch_avg_bbox) / pytorch_avg_bbox * 100
                f.write(f"| 平均边界框损失 | {jittor_avg_bbox:.4f} | {pytorch_avg_bbox:.4f} | {diff_pct:.2f}% |\n")
            else:
                f.write("| 平均边界框损失 | 数据不可用 | 数据不可用 | - |\n")
            
            if jittor_metrics['loss_dfl'] and pytorch_metrics['loss_dfl']:
                jittor_avg_dfl = np.mean(jittor_metrics['loss_dfl'])
                pytorch_avg_dfl = np.mean(pytorch_metrics['loss_dfl'])
                diff_pct = abs(jittor_avg_dfl - pytorch_avg_dfl) / pytorch_avg_dfl * 100
                f.write(f"| 平均DFL损失 | {jittor_avg_dfl:.4f} | {pytorch_avg_dfl:.4f} | {diff_pct:.2f}% |\n\n")
            else:
                f.write("| 平均DFL损失 | 数据不可用 | 数据不可用 | - |\n\n")
        else:
            f.write("| 平均总损失 | 数据不可用 | 数据不可用 | - |\n")
            f.write("| 平均分类损失 | 数据不可用 | 数据不可用 | - |\n")
            f.write("| 平均边界框损失 | 数据不可用 | 数据不可用 | - |\n")
            f.write("| 平均DFL损失 | 数据不可用 | 数据不可用 | - |\n\n")
            f.write("\n> **注意**: 日志中未找到完整的损失记录，建议修改训练代码添加详细的损失打印。\n\n")
        
        # 训练性能对比
        f.write("## 训练性能对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 比例 |\n")
        f.write("|------|--------|---------|------|\n")
        
        if jittor_avg_time is not None and pytorch_avg_time is not None:
            speedup = pytorch_avg_time / jittor_avg_time
            f.write(f"| 平均每次迭代时间 | {jittor_avg_time:.4f}秒 | {pytorch_avg_time:.4f}秒 | {speedup:.2f}x |\n\n")
        else:
            f.write("| 平均每次迭代时间 | 数据不可用 | 数据不可用 | - |\n\n")
            f.write("\n> **注意**: 日志中未找到完整的时间记录，建议修改训练代码添加详细的时间打印。\n\n")
        
        # 评估结果对比
        f.write("## 评估结果对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 差异比例 |\n")
        f.write("|------|--------|---------|----------|\n")
        
        # 评估指标对比表格
        metrics_to_compare = [
            ('mAP', 'mAP'),
            ('mAP_50', 'mAP@0.5'),
            ('mAP_75', 'mAP@0.75'),
            ('mAP_s', 'mAP (small)'),
            ('mAP_m', 'mAP (medium)'),
            ('mAP_l', 'mAP (large)')
        ]
        
        has_valid_metrics = False
        for metric_key, metric_name in metrics_to_compare:
            jittor_value = jittor_eval.get(metric_key)
            pytorch_value = pytorch_eval.get(metric_key)
            
            if jittor_value is not None and pytorch_value is not None:
                has_valid_metrics = True
                if pytorch_value == 0:
                    # 避免除零错误
                    diff_pct = 'N/A' if jittor_value == 0 else 'inf'
                else:
                    diff_pct = abs(jittor_value - pytorch_value) / pytorch_value * 100
                    diff_pct = f"{diff_pct:.2f}%"
                
                f.write(f"| {metric_name} | {jittor_value:.4f} | {pytorch_value:.4f} | {diff_pct} |\n")
            else:
                jittor_str = f"{jittor_value:.4f}" if jittor_value is not None else "数据不可用"
                pytorch_str = f"{pytorch_value:.4f}" if pytorch_value is not None else "数据不可用"
                f.write(f"| {metric_name} | {jittor_str} | {pytorch_str} | - |\n")
        
        if not has_valid_metrics:
            f.write("\n> **注意**: 日志中未找到完整的评估指标记录，可能是因为测试结果为空。建议重新运行测试。\n\n")
            
        # 损失曲线
        if jittor_metrics['loss'] and pytorch_metrics['loss']:
            f.write("\n## 损失曲线\n\n")
            f.write("![Loss Comparison](loss_comparison.png)\n\n")
        
        # 训练时间
        if jittor_metrics['time'] and pytorch_metrics['time']:
            f.write("## 训练时间\n\n")
            f.write("![Time Comparison](time_comparison.png)\n\n")
        
        # mAP对比图
        if has_valid_metrics:
            f.write("## 评估指标对比图\n\n")
            f.write("![mAP Comparison](map_comparison.png)\n\n")
        
        # 结论
        f.write("## 结论\n\n")
        
        # 损失函数对齐分析
        f.write("### 1. 损失函数对齐情况\n\n")
        if jittor_avg_loss is not None and pytorch_avg_loss is not None:
            diff_pct = abs(jittor_avg_loss - pytorch_avg_loss) / pytorch_avg_loss * 100
            if diff_pct < 5:
                f.write("两个框架的损失函数对齐良好，总损失差异小于5%。")
            elif diff_pct < 10:
                f.write(f"两个框架的损失函数存在一定差异，总损失差异约为{diff_pct:.2f}%，但在可接受范围内。")
            else:
                f.write(f"两个框架的损失函数存在明显差异，总损失差异达{diff_pct:.2f}%，需要进一步调查原因。")
            
            # 分析各部分损失
            if jittor_metrics['loss_cls'] and pytorch_metrics['loss_cls'] and \
               jittor_metrics['loss_bbox'] and pytorch_metrics['loss_bbox'] and \
               jittor_metrics['loss_dfl'] and pytorch_metrics['loss_dfl']:
                
                jittor_avg_cls = np.mean(jittor_metrics['loss_cls'])
                pytorch_avg_cls = np.mean(pytorch_metrics['loss_cls'])
                cls_diff_pct = abs(jittor_avg_cls - pytorch_avg_cls) / pytorch_avg_cls * 100
                
                jittor_avg_bbox = np.mean(jittor_metrics['loss_bbox'])
                pytorch_avg_bbox = np.mean(pytorch_metrics['loss_bbox'])
                bbox_diff_pct = abs(jittor_avg_bbox - pytorch_avg_bbox) / pytorch_avg_bbox * 100
                
                jittor_avg_dfl = np.mean(jittor_metrics['loss_dfl'])
                pytorch_avg_dfl = np.mean(pytorch_metrics['loss_dfl'])
                dfl_diff_pct = abs(jittor_avg_dfl - pytorch_avg_dfl) / pytorch_avg_dfl * 100
                
                high_diff_components = []
                if cls_diff_pct > 10:
                    high_diff_components.append(f"分类损失(差异{cls_diff_pct:.2f}%)")
                if bbox_diff_pct > 10:
                    high_diff_components.append(f"边界框损失(差异{bbox_diff_pct:.2f}%)")
                if dfl_diff_pct > 10:
                    high_diff_components.append(f"DFL损失(差异{dfl_diff_pct:.2f}%)")
                
                if high_diff_components:
                    f.write(f" 其中，{', '.join(high_diff_components)}的差异较大，可能需要重点关注这些组件的实现。")
                else:
                    f.write(" 各损失组件的对齐情况都很好。")
            
            f.write("\n\n")
        else:
            f.write("日志中未找到完整的损失记录，无法评估损失函数的对齐情况。建议修改训练代码，添加详细的损失打印。\n\n")
            
        # 训练性能分析
        f.write("### 2. 训练性能\n\n")
        if jittor_avg_time is not None and pytorch_avg_time is not None:
            speedup = pytorch_avg_time / jittor_avg_time
            if speedup > 1:
                f.write(f"Jittor框架的训练速度比PyTorch快{speedup:.2f}倍。")
            elif speedup < 1:
                f.write(f"PyTorch框架的训练速度比Jittor快{1/speedup:.2f}倍。")
            else:
                f.write("两个框架的训练速度基本相同。")
                
            if speedup > 1.5 or speedup < 0.67:
                f.write(" 两个框架的性能差异明显，建议进一步分析是否有优化空间。")
            f.write("\n\n")
        else:
            f.write("日志中未找到完整的时间记录，无法评估训练性能。建议修改训练代码，添加详细的时间打印。\n\n")
        
        # 评估结果分析
        f.write("### 3. 评估结果\n\n")
        if has_valid_metrics:
            jittor_map = jittor_eval.get('mAP')
            pytorch_map = pytorch_eval.get('mAP')
            if jittor_map is not None and pytorch_map is not None:
                if pytorch_map == 0:
                    if jittor_map == 0:
                        f.write("两个框架的mAP评估结果均为0，可能是因为训练轮数不足或数据集问题。")
                    else:
                        f.write(f"PyTorch框架的mAP为0，而Jittor框架的mAP为{jittor_map:.4f}，存在明显差异，需要进一步调查。")
                else:
                    diff_pct = abs(jittor_map - pytorch_map) / pytorch_map * 100
                    if diff_pct < 5:
                        f.write(f"两个框架的评估结果对齐良好，mAP差异仅为{diff_pct:.2f}%。")
                    elif diff_pct < 10:
                        f.write(f"两个框架的评估结果存在一定差异，mAP差异为{diff_pct:.2f}%，但在可接受范围内。")
                    else:
                        f.write(f"两个框架的评估结果存在明显差异，mAP差异达{diff_pct:.2f}%，需要进一步调查原因。")
                
                # 分析不同尺寸物体的mAP
                jittor_map_s = jittor_eval.get('mAP_s')
                pytorch_map_s = pytorch_eval.get('mAP_s')
                jittor_map_m = jittor_eval.get('mAP_m')
                pytorch_map_m = pytorch_eval.get('mAP_m')
                jittor_map_l = jittor_eval.get('mAP_l')
                pytorch_map_l = pytorch_eval.get('mAP_l')
                
                size_diff = []
                if jittor_map_s is not None and pytorch_map_s is not None and pytorch_map_s != 0:
                    s_diff_pct = abs(jittor_map_s - pytorch_map_s) / pytorch_map_s * 100
                    if s_diff_pct > 10:
                        size_diff.append(f"小物体(差异{s_diff_pct:.2f}%)")
                
                if jittor_map_m is not None and pytorch_map_m is not None and pytorch_map_m != 0:
                    m_diff_pct = abs(jittor_map_m - pytorch_map_m) / pytorch_map_m * 100
                    if m_diff_pct > 10:
                        size_diff.append(f"中物体(差异{m_diff_pct:.2f}%)")
                
                if jittor_map_l is not None and pytorch_map_l is not None and pytorch_map_l != 0:
                    l_diff_pct = abs(jittor_map_l - pytorch_map_l) / pytorch_map_l * 100
                    if l_diff_pct > 10:
                        size_diff.append(f"大物体(差异{l_diff_pct:.2f}%)")
                
                if size_diff:
                    f.write(f" 其中，{', '.join(size_diff)}的检测精度差异较大，可能需要重点关注这些尺寸物体的检测实现。")
            
            f.write("\n\n")
        else:
            f.write("日志中未找到完整的评估指标记录，无法评估检测精度。可能是因为测试结果为空，建议重新运行测试。\n\n")
        
        # 总结
        f.write("### 4. 总体结论\n\n")
        if jittor_avg_loss is not None and pytorch_avg_loss is not None and has_valid_metrics:
            loss_aligned = abs(jittor_avg_loss - pytorch_avg_loss) / pytorch_avg_loss * 100 < 10
            
            jittor_map = jittor_eval.get('mAP')
            pytorch_map = pytorch_eval.get('mAP')
            map_aligned = False
            if jittor_map is not None and pytorch_map is not None and pytorch_map != 0:
                map_aligned = abs(jittor_map - pytorch_map) / pytorch_map * 100 < 10
            
            if loss_aligned and map_aligned:
                f.write("总体而言，Jittor和PyTorch框架下的GFL实现对齐良好，两个框架的损失函数和评估结果差异均在可接受范围内。")
            elif loss_aligned:
                f.write("Jittor和PyTorch框架下的GFL实现在损失函数方面对齐良好，但评估结果存在一定差异，可能需要进一步调整模型或训练参数。")
            elif map_aligned:
                f.write("Jittor和PyTorch框架下的GFL实现在评估结果方面对齐良好，但损失函数存在一定差异，可能需要检查损失计算的实现细节。")
            else:
                f.write("Jittor和PyTorch框架下的GFL实现在损失函数和评估结果方面均存在一定差异，建议进一步分析两个框架的实现差异，并调整相关参数。")
        else:
            f.write("由于日志数据不完整，无法做出全面的对齐结论。建议修改训练代码添加更详细的日志记录，并重新运行测试，以便进行更准确的对比分析。")
    
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
        
        # 生成mAP比较图表
        if jittor_eval and pytorch_eval:
            generate_map_comparison_plot(jittor_eval, pytorch_eval, output_dir)
        
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