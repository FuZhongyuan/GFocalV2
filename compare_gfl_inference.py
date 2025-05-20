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
        logging.FileHandler("inference_comparison.log"),
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

class InferenceLogParser:
    """解析推理日志并提取关键信息"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics = {
            'inference_time': [],  # 推理时间列表 (每张图片)
            'fps': None,           # 帧率 (FPS)
            'bbox_mAP': None,      # mAP
            'bbox_mAP_50': None,   # mAP@0.5
            'bbox_mAP_75': None,   # mAP@0.75
            'bbox_mAP_s': None,    # small objects mAP
            'bbox_mAP_m': None,    # medium objects mAP
            'bbox_mAP_l': None,    # large objects mAP
        }
        
    def parse_jittor_log(self):
        """解析Jittor框架的推理日志文件"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取推理时间
        time_pattern = r"time: ([\d\.]+) ms"
        time_matches = re.findall(time_pattern, content)
        for match in time_matches:
            self.metrics['inference_time'].append(float(match))
        
        if self.metrics['inference_time']:
            # 计算FPS (帧每秒)
            avg_inference_time = np.mean(self.metrics['inference_time'])
            self.metrics['fps'] = 1000 / avg_inference_time
            
        # 提取评估结果
        # COCO标准评估指标
        ap_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        ap_50_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        ap_75_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.75\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        ap_s_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=small\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        ap_m_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=medium\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        ap_l_pattern = r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0.50:0.95\s+\|\s+area=large\s+\|\s+maxDets=100\s+\]\s+=\s+([\d\.]+)"
        
        ap_match = re.search(ap_pattern, content)
        ap_50_match = re.search(ap_50_pattern, content)
        ap_75_match = re.search(ap_75_pattern, content)
        ap_s_match = re.search(ap_s_pattern, content)
        ap_m_match = re.search(ap_m_pattern, content)
        ap_l_match = re.search(ap_l_pattern, content)
        
        if ap_match:
            self.metrics['bbox_mAP'] = float(ap_match.group(1))
        if ap_50_match:
            self.metrics['bbox_mAP_50'] = float(ap_50_match.group(1))
        if ap_75_match:
            self.metrics['bbox_mAP_75'] = float(ap_75_match.group(1))
        if ap_s_match:
            self.metrics['bbox_mAP_s'] = float(ap_s_match.group(1))
        if ap_m_match:
            self.metrics['bbox_mAP_m'] = float(ap_m_match.group(1))
        if ap_l_match:
            self.metrics['bbox_mAP_l'] = float(ap_l_match.group(1))
            
        return self.metrics
    
    def parse_pytorch_log(self):
        """解析PyTorch框架的推理日志文件"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取推理时间
        time_pattern = r"time: ([\d\.]+) ms"
        time_matches = re.findall(time_pattern, content)
        for match in time_matches:
            self.metrics['inference_time'].append(float(match))
        
        if self.metrics['inference_time']:
            # 计算FPS (帧每秒)
            avg_inference_time = np.mean(self.metrics['inference_time'])
            self.metrics['fps'] = 1000 / avg_inference_time
        
        # 提取评估结果 (MMDetection输出格式)
        bbox_map_pattern = r"bbox_mAP: ([\d\.]+)"
        bbox_map_50_pattern = r"bbox_mAP_50: ([\d\.]+)"
        bbox_map_75_pattern = r"bbox_mAP_75: ([\d\.]+)"
        bbox_map_s_pattern = r"bbox_mAP_s: ([\d\.]+)"
        bbox_map_m_pattern = r"bbox_mAP_m: ([\d\.]+)"
        bbox_map_l_pattern = r"bbox_mAP_l: ([\d\.]+)"
        
        bbox_map_match = re.search(bbox_map_pattern, content)
        bbox_map_50_match = re.search(bbox_map_50_pattern, content)
        bbox_map_75_match = re.search(bbox_map_75_pattern, content)
        bbox_map_s_match = re.search(bbox_map_s_pattern, content)
        bbox_map_m_match = re.search(bbox_map_m_pattern, content)
        bbox_map_l_match = re.search(bbox_map_l_pattern, content)
        
        if bbox_map_match:
            self.metrics['bbox_mAP'] = float(bbox_map_match.group(1))
        if bbox_map_50_match:
            self.metrics['bbox_mAP_50'] = float(bbox_map_50_match.group(1))
        if bbox_map_75_match:
            self.metrics['bbox_mAP_75'] = float(bbox_map_75_match.group(1))
        if bbox_map_s_match:
            self.metrics['bbox_mAP_s'] = float(bbox_map_s_match.group(1))
        if bbox_map_m_match:
            self.metrics['bbox_mAP_m'] = float(bbox_map_m_match.group(1))
        if bbox_map_l_match:
            self.metrics['bbox_mAP_l'] = float(bbox_map_l_match.group(1))
            
        return self.metrics


def find_latest_checkpoint(framework):
    """查找最新的检查点文件"""
    if framework == 'jittor':
        checkpoint_dir = os.path.join(JITTOR_WORKDIR, "work_dirs")
        pattern = "*.pkl"
    else:  # pytorch
        checkpoint_dir = os.path.join(PYTORCH_WORKDIR, "work_dirs")
        pattern = "*.pth"
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"找不到{framework}框架的检查点目录: {checkpoint_dir}")
        return None
    
    checkpoints = []
    # 遍历所有子目录查找检查点
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".pkl"):
                checkpoints.append(os.path.join(root, file))
    
    # 按文件修改时间排序
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if checkpoints:
        return checkpoints[0]
    else:
        logger.error(f"在{checkpoint_dir}中找不到任何{framework}框架的检查点")
        return None


def run_inference(framework, checkpoint=None, batch_size=1, num_samples=None):
    """运行指定框架的推理过程"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(framework)
        if checkpoint is None:
            logger.error(f"未能找到{framework}框架的检查点，无法进行推理")
            return None
    
    if framework == 'jittor':
        config = JITTOR_CONFIG
        work_dir = f"infer_results/gfl_jittor_{timestamp}"
        cmd = f"cd {JITTOR_WORKDIR} && pip install -e .&& python tools/test.py {config} {checkpoint} --eval bbox"
        if batch_size != 1:
            cmd += f" --batch-size {batch_size}"
        if num_samples:
            cmd += f" --samples {num_samples}"
        log_file = os.path.join(JITTOR_WORKDIR, work_dir, "inference.log")
    else:  # pytorch
        config = PYTORCH_CONFIG
        work_dir = f"infer_results/gfl_pytorch_{timestamp}"
        cmd = f"cd {PYTORCH_WORKDIR} && pip install -e .&& python tools/test.py {config} {checkpoint} --eval bbox"
        if batch_size != 1:
            cmd += f" --batch-size {batch_size}"
        if num_samples:
            cmd += f" --samples {num_samples}"
        log_file = os.path.join(PYTORCH_WORKDIR, work_dir, "inference.log")
    
    # 创建工作目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.info(f"开始在 {framework} 框架下运行推理测试...")
    logger.info(f"使用检查点: {checkpoint}")
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
    logger.info(f"{framework} 推理测试完成，返回码: {return_code}")
    
    if return_code != 0:
        logger.error(f"{framework} 推理测试失败，请检查日志: {log_file}")
        return None
    
    return log_file


def generate_inference_time_plot(jittor_times, pytorch_times, output_dir):
    """生成推理时间对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # 直方图对比
    bins = np.linspace(
        min(min(jittor_times), min(pytorch_times)), 
        max(max(jittor_times), max(pytorch_times)), 
        30
    )
    plt.hist(jittor_times, bins=bins, alpha=0.5, label='Jittor')
    plt.hist(pytorch_times, bins=bins, alpha=0.5, label='PyTorch')
    
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Inference Time Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
    logger.info(f"推理时间对比图已保存到 {os.path.join(output_dir, 'inference_time_comparison.png')}")
    
    # 箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot([jittor_times, pytorch_times], labels=['Jittor', 'PyTorch'])
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_boxplot.png'))
    logger.info(f"推理时间箱线图已保存到 {os.path.join(output_dir, 'inference_time_boxplot.png')}")


def generate_ap_comparison_plot(jittor_metrics, pytorch_metrics, output_dir):
    """生成mAP对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l']
    metric_names = ['mAP', 'mAP@0.5', 'mAP@0.75', 'mAP (small)', 'mAP (medium)', 'mAP (large)']
    
    # 筛选有效的指标
    valid_metrics = []
    jittor_values = []
    pytorch_values = []
    labels = []
    
    for i, metric in enumerate(metrics_to_plot):
        if jittor_metrics[metric] is not None and pytorch_metrics[metric] is not None:
            valid_metrics.append(metric)
            jittor_values.append(jittor_metrics[metric])
            pytorch_values.append(pytorch_metrics[metric])
            labels.append(metric_names[i])
    
    # 如果没有有效的指标，则退出
    if not valid_metrics:
        logger.warning("没有找到有效的评估指标，无法生成对比图")
        return
    
    # 创建柱状图
    x = np.arange(len(valid_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, jittor_values, width, label='Jittor')
    rects2 = ax.bar(x + width/2, pytorch_values, width, label='PyTorch')
    
    ax.set_ylabel('Score')
    ax.set_title('Detection Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_comparison.png'))
    logger.info(f"mAP对比图已保存到 {os.path.join(output_dir, 'ap_comparison.png')}")


def generate_report(jittor_metrics, pytorch_metrics, output_dir):
    """生成推理对比报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'inference_comparison_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# GFL在Jittor和PyTorch框架下的推理性能对比报告\n\n")
        f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 推理配置\n\n")
        f.write(f"Jittor配置文件: `{JITTOR_CONFIG}`\n\n")
        f.write(f"PyTorch配置文件: `{PYTORCH_CONFIG}`\n\n")
        
        f.write("## 推理性能对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 比例 |\n")
        f.write("|------|--------|---------|------|\n")
        
        # 计算平均推理时间和FPS
        jittor_avg_time = np.mean(jittor_metrics['inference_time']) if jittor_metrics['inference_time'] else None
        pytorch_avg_time = np.mean(pytorch_metrics['inference_time']) if pytorch_metrics['inference_time'] else None
        
        if jittor_avg_time and pytorch_avg_time:
            speedup = pytorch_avg_time / jittor_avg_time
            f.write(f"| 平均推理时间 | {jittor_avg_time:.2f} ms | {pytorch_avg_time:.2f} ms | {speedup:.2f}x |\n")
            f.write(f"| FPS (帧每秒) | {jittor_metrics['fps']:.2f} | {pytorch_metrics['fps']:.2f} | {speedup:.2f}x |\n")
        
        f.write("\n## 检测精度对比\n\n")
        f.write("| 指标 | Jittor | PyTorch | 差异比例 |\n")
        f.write("|------|--------|---------|----------|\n")
        
        # 比较mAP指标
        metrics_to_compare = [
            ('bbox_mAP', 'mAP'),
            ('bbox_mAP_50', 'mAP@0.5'),
            ('bbox_mAP_75', 'mAP@0.75'),
            ('bbox_mAP_s', 'mAP (small)'),
            ('bbox_mAP_m', 'mAP (medium)'),
            ('bbox_mAP_l', 'mAP (large)')
        ]
        
        for metric_key, metric_name in metrics_to_compare:
            jittor_value = jittor_metrics[metric_key]
            pytorch_value = pytorch_metrics[metric_key]
            
            if jittor_value is not None and pytorch_value is not None:
                diff_pct = abs(jittor_value - pytorch_value) / pytorch_value * 100
                f.write(f"| {metric_name} | {jittor_value:.4f} | {pytorch_value:.4f} | {diff_pct:.2f}% |\n")
        
        f.write("\n## 推理时间分布\n\n")
        f.write("![Inference Time Comparison](inference_time_comparison.png)\n\n")
        f.write("![Inference Time Boxplot](inference_time_boxplot.png)\n\n")
        
        f.write("## 检测精度对比\n\n")
        f.write("![AP Comparison](ap_comparison.png)\n\n")
        
        f.write("## 结论\n\n")
        
        # 性能结论
        f.write("1. **推理性能**: ")
        if jittor_avg_time and pytorch_avg_time:
            if speedup > 1:
                f.write(f"Jittor框架的推理速度比PyTorch快{speedup:.2f}倍。\n")
            else:
                f.write(f"PyTorch框架的推理速度比Jittor快{1/speedup:.2f}倍。\n")
        else:
            f.write("未能获取完整的推理时间数据，无法比较性能。\n")
        
        # 准确率结论
        f.write("2. **检测精度**: ")
        if jittor_metrics['bbox_mAP'] is not None and pytorch_metrics['bbox_mAP'] is not None:
            diff_pct = abs(jittor_metrics['bbox_mAP'] - pytorch_metrics['bbox_mAP']) / pytorch_metrics['bbox_mAP'] * 100
            if diff_pct < 5:
                f.write(f"两个框架的检测精度对齐良好，mAP差异仅为{diff_pct:.2f}%。\n")
            else:
                f.write(f"两个框架的检测精度存在一定差异，mAP差异为{diff_pct:.2f}%。\n")
        else:
            f.write("未能获取完整的mAP数据，无法比较检测精度。\n")
    
    logger.info(f"推理对比报告已保存到 {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description='对比PyTorch和Jittor框架下的GFL推理性能')
    parser.add_argument('--jittor-checkpoint', type=str, help='Jittor框架的检查点文件路径')
    parser.add_argument('--pytorch-checkpoint', type=str, help='PyTorch框架的检查点文件路径')
    parser.add_argument('--batch-size', type=int, default=1, help='推理批次大小')
    parser.add_argument('--num-samples', type=int, help='测试样本数量')
    parser.add_argument('--output-dir', type=str, default='inference_comparison_results', help='结果输出目录')
    parser.add_argument('--jittor-only', action='store_true', help='仅测试Jittor框架')
    parser.add_argument('--pytorch-only', action='store_true', help='仅测试PyTorch框架')
    parser.add_argument('--jittor-log', type=str, help='指定Jittor推理日志文件(不运行推理)')
    parser.add_argument('--pytorch-log', type=str, help='指定PyTorch推理日志文件(不运行推理)')
    
    args = parser.parse_args()
    
    logger.info("开始GFL框架推理性能对比测试")
    logger.info(f"参数: {args}")
    
    jittor_log_file = None
    pytorch_log_file = None
    
    # 运行推理或加载指定的日志文件
    if args.jittor_log:
        jittor_log_file = args.jittor_log
        logger.info(f"使用指定的Jittor推理日志文件: {jittor_log_file}")
    elif not args.pytorch_only:
        jittor_log_file = run_inference(
            'jittor', 
            checkpoint=args.jittor_checkpoint, 
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
    
    if args.pytorch_log:
        pytorch_log_file = args.pytorch_log
        logger.info(f"使用指定的PyTorch推理日志文件: {pytorch_log_file}")
    elif not args.jittor_only:
        pytorch_log_file = run_inference(
            'pytorch', 
            checkpoint=args.pytorch_checkpoint, 
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
    
    # 解析日志
    if jittor_log_file:
        jittor_parser = InferenceLogParser(jittor_log_file)
        jittor_metrics = jittor_parser.parse_jittor_log()
        logger.info(f"成功解析Jittor推理日志，提取了{len(jittor_metrics['inference_time'])}条推理时间记录")
    else:
        jittor_metrics = None
    
    if pytorch_log_file:
        pytorch_parser = InferenceLogParser(pytorch_log_file)
        pytorch_metrics = pytorch_parser.parse_pytorch_log()
        logger.info(f"成功解析PyTorch推理日志，提取了{len(pytorch_metrics['inference_time'])}条推理时间记录")
    else:
        pytorch_metrics = None
    
    # 生成对比报告
    if jittor_metrics and pytorch_metrics:
        output_dir = os.path.abspath(args.output_dir)
        
        # 生成推理时间对比图
        if jittor_metrics['inference_time'] and pytorch_metrics['inference_time']:
            generate_inference_time_plot(
                jittor_metrics['inference_time'], 
                pytorch_metrics['inference_time'],
                output_dir
            )
        
        # 生成mAP对比图
        generate_ap_comparison_plot(jittor_metrics, pytorch_metrics, output_dir)
        
        # 生成报告
        report_file = generate_report(jittor_metrics, pytorch_metrics, output_dir)
        logger.info(f"推理对比完成! 报告保存在: {report_file}")
    elif jittor_metrics:
        logger.info("只有Jittor框架的结果，无法生成对比报告")
    elif pytorch_metrics:
        logger.info("只有PyTorch框架的结果，无法生成对比报告")
    else:
        logger.error("没有任何框架的结果，无法生成报告")

if __name__ == '__main__':
    main()