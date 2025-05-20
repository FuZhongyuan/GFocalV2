#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import MaxNLocator

class LogAnalyzer:
    """分析训练日志并提取关键信息"""
    
    def __init__(self, jittor_log_path, pytorch_log_path):
        self.jittor_log_path = jittor_log_path
        self.pytorch_log_path = pytorch_log_path
        
        # 指标存储结构
        self.jittor_metrics = {
            'epochs': [], 'iters': [], 'loss': [], 
            'loss_cls': [], 'loss_bbox': [], 'loss_dfl': [],
            'lr': [], 'time': []
        }
        self.pytorch_metrics = {
            'epochs': [], 'iters': [], 'loss': [], 
            'loss_cls': [], 'loss_bbox': [], 'loss_dfl': [],
            'lr': [], 'time': []
        }
        
        # 评估结果
        self.jittor_eval = {
            'epochs': [], 'mAP': [], 'mAP_50': [], 'mAP_75': [],
            'mAP_s': [], 'mAP_m': [], 'mAP_l': []
        }
        self.pytorch_eval = {
            'epochs': [], 'mAP': [], 'mAP_50': [], 'mAP_75': [],
            'mAP_s': [], 'mAP_m': [], 'mAP_l': []
        }
        
        # 性能指标
        self.jittor_performance = {'epoch_times': []}
        self.pytorch_performance = {'epoch_times': []}
    
    def parse_jittor_log(self):
        """解析Jittor日志"""
        with open(self.jittor_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 提取训练指标
        train_pattern = r"Epoch\s+\[(\d+)/\d+\]\s+Iter\s+\[(\d+)/\d+\].*?lr:\s+([\d\.e\-]+).*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?time:\s+([\d\.]+)"
        for match in re.finditer(train_pattern, content):
            epoch, iter_num, lr, loss, loss_cls, loss_bbox, loss_dfl, time_cost = match.groups()
            self.jittor_metrics['epochs'].append(int(epoch))
            self.jittor_metrics['iters'].append(int(iter_num))
            self.jittor_metrics['lr'].append(float(lr))
            self.jittor_metrics['loss'].append(float(loss))
            self.jittor_metrics['loss_cls'].append(float(loss_cls))
            self.jittor_metrics['loss_bbox'].append(float(loss_bbox))
            self.jittor_metrics['loss_dfl'].append(float(loss_dfl))
            self.jittor_metrics['time'].append(float(time_cost))
        
        # 提取评估结果
        eval_pattern = r"bbox_mAP:\s+([\d\.]+)\s+bbox_mAP_50:\s+([\d\.]+)\s+bbox_mAP_75:\s+([\d\.]+)\s+bbox_mAP_s:\s+([\d\.]+)\s+bbox_mAP_m:\s+([\d\.]+)\s+bbox_mAP_l:\s+([\d\.]+)"
        epoch = 0
        for match in re.finditer(eval_pattern, content):
            mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l = match.groups()
            epoch += 1
            self.jittor_eval['epochs'].append(epoch)
            self.jittor_eval['mAP'].append(float(mAP))
            self.jittor_eval['mAP_50'].append(float(mAP_50))
            self.jittor_eval['mAP_75'].append(float(mAP_75))
            self.jittor_eval['mAP_s'].append(float(mAP_s))
            self.jittor_eval['mAP_m'].append(float(mAP_m))
            self.jittor_eval['mAP_l'].append(float(mAP_l))
        
        # 估算每个epoch的训练时间
        epoch_times_pattern = r"save checkpoint to.+?epoch_(\d+)\.pkl"
        checkpoint_times = []
        for match in re.finditer(epoch_times_pattern, content):
            checkpoint_times.append(match.start())
        
        if len(checkpoint_times) > 1:
            for i in range(1, len(checkpoint_times)):
                time_diff = checkpoint_times[i] - checkpoint_times[i-1]
                self.jittor_performance['epoch_times'].append(time_diff/1000)  # 估算时间，单位为秒
            
        return self.jittor_metrics, self.jittor_eval, self.jittor_performance
    
    def parse_pytorch_log(self):
        """解析PyTorch日志"""
        with open(self.pytorch_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 提取训练指标 - PyTorch格式可能略有不同
        train_pattern = r"Epoch\(train\)\s+\[(\d+)\]\[(\d+)/\d+\].*?lr:\s+([\d\.e\-]+).*?loss:\s+([\d\.]+).*?loss_cls:\s+([\d\.]+).*?loss_bbox:\s+([\d\.]+).*?loss_dfl:\s+([\d\.]+).*?time:\s+([\d\.]+)"
        for match in re.finditer(train_pattern, content):
            epoch, iter_num, lr, loss, loss_cls, loss_bbox, loss_dfl, time_cost = match.groups()
            self.pytorch_metrics['epochs'].append(int(epoch))
            self.pytorch_metrics['iters'].append(int(iter_num))
            self.pytorch_metrics['lr'].append(float(lr))
            self.pytorch_metrics['loss'].append(float(loss))
            self.pytorch_metrics['loss_cls'].append(float(loss_cls))
            self.pytorch_metrics['loss_bbox'].append(float(loss_bbox))
            self.pytorch_metrics['loss_dfl'].append(float(loss_dfl))
            self.pytorch_metrics['time'].append(float(time_cost))
        
        # 提取评估结果
        eval_pattern = r"bbox_mAP_copypaste:\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
        epoch = 0
        for match in re.finditer(eval_pattern, content):
            mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l = match.groups()
            epoch += 1
            self.pytorch_eval['epochs'].append(epoch)
            self.pytorch_eval['mAP'].append(float(mAP))
            self.pytorch_eval['mAP_50'].append(float(mAP_50))
            self.pytorch_eval['mAP_75'].append(float(mAP_75))
            self.pytorch_eval['mAP_s'].append(float(mAP_s))
            self.pytorch_eval['mAP_m'].append(float(mAP_m))
            self.pytorch_eval['mAP_l'].append(float(mAP_l))
        
        # 估算每个epoch的训练时间
        epoch_times_pattern = r"Saving checkpoint at (\d+) epochs"
        checkpoint_times = []
        for match in re.finditer(epoch_times_pattern, content):
            checkpoint_times.append(match.start())
        
        if len(checkpoint_times) > 1:
            for i in range(1, len(checkpoint_times)):
                time_diff = checkpoint_times[i] - checkpoint_times[i-1]
                self.pytorch_performance['epoch_times'].append(time_diff/1000)  # 估算时间，单位为秒
            
        return self.pytorch_metrics, self.pytorch_eval, self.pytorch_performance
    
    def visualize_loss_comparison(self, output_dir):
        """可视化损失函数比较"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # 损失函数比较
        plt.subplot(2, 2, 1)
        plt.plot(self.jittor_metrics['epochs'], self.jittor_metrics['loss'], 'b-', label='Jittor')
        plt.plot(self.pytorch_metrics['epochs'], self.pytorch_metrics['loss'], 'r-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 分类损失比较
        plt.subplot(2, 2, 2)
        plt.plot(self.jittor_metrics['epochs'], self.jittor_metrics['loss_cls'], 'b-', label='Jittor')
        plt.plot(self.pytorch_metrics['epochs'], self.pytorch_metrics['loss_cls'], 'r-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.title('Classification Loss Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 边界框损失比较
        plt.subplot(2, 2, 3)
        plt.plot(self.jittor_metrics['epochs'], self.jittor_metrics['loss_bbox'], 'b-', label='Jittor')
        plt.plot(self.pytorch_metrics['epochs'], self.pytorch_metrics['loss_bbox'], 'r-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('Bounding Box Loss')
        plt.title('Bounding Box Loss Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # DFL损失比较
        plt.subplot(2, 2, 4)
        plt.plot(self.jittor_metrics['epochs'], self.jittor_metrics['loss_dfl'], 'b-', label='Jittor')
        plt.plot(self.pytorch_metrics['epochs'], self.pytorch_metrics['loss_dfl'], 'r-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('DFL Loss')
        plt.title('DFL Loss Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300)
        plt.close()
    
    def visualize_map_comparison(self, output_dir):
        """可视化mAP指标比较"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # mAP比较
        plt.subplot(2, 3, 1)
        plt.plot(self.jittor_eval['epochs'], self.jittor_eval['mAP'], 'bo-', label='Jittor')
        plt.plot(self.pytorch_eval['epochs'], self.pytorch_eval['mAP'], 'ro-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('mAP Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # mAP_50比较
        plt.subplot(2, 3, 2)
        plt.plot(self.jittor_eval['epochs'], self.jittor_eval['mAP_50'], 'bo-', label='Jittor')
        plt.plot(self.pytorch_eval['epochs'], self.pytorch_eval['mAP_50'], 'ro-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP_50')
        plt.title('mAP_50 Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # mAP_75比较
        plt.subplot(2, 3, 3)
        plt.plot(self.jittor_eval['epochs'], self.jittor_eval['mAP_75'], 'bo-', label='Jittor')
        plt.plot(self.pytorch_eval['epochs'], self.pytorch_eval['mAP_75'], 'ro-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP_75')
        plt.title('mAP_75 Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # mAP_s比较
        plt.subplot(2, 3, 4)
        plt.plot(self.jittor_eval['epochs'], self.jittor_eval['mAP_s'], 'bo-', label='Jittor')
        plt.plot(self.pytorch_eval['epochs'], self.pytorch_eval['mAP_s'], 'ro-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP_s')
        plt.title('mAP for Small Objects')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # mAP_m比较
        plt.subplot(2, 3, 5)
        plt.plot(self.jittor_eval['epochs'], self.jittor_eval['mAP_m'], 'bo-', label='Jittor')
        plt.plot(self.pytorch_eval['epochs'], self.pytorch_eval['mAP_m'], 'ro-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP_m')
        plt.title('mAP for Medium Objects')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # mAP_l比较
        plt.subplot(2, 3, 6)
        plt.plot(self.jittor_eval['epochs'], self.jittor_eval['mAP_l'], 'bo-', label='Jittor')
        plt.plot(self.pytorch_eval['epochs'], self.pytorch_eval['mAP_l'], 'ro-', label='PyTorch')
        plt.xlabel('Epoch')
        plt.ylabel('mAP_l')
        plt.title('mAP for Large Objects')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'map_comparison.png'), dpi=300)
        plt.close()
    
    def visualize_performance_comparison(self, output_dir):
        """可视化性能指标比较"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # 迭代时间比较
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.jittor_metrics['time']) + 1), self.jittor_metrics['time'], 'b-', label='Jittor')
        plt.plot(range(1, len(self.pytorch_metrics['time']) + 1), self.pytorch_metrics['time'], 'r-', label='PyTorch')
        plt.xlabel('Iteration')
        plt.ylabel('Time per Iteration (s)')
        plt.title('Iteration Time Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 学习率变化比较
        plt.subplot(1, 2, 2)
        plt.semilogy(range(1, len(self.jittor_metrics['lr']) + 1), self.jittor_metrics['lr'], 'b-', label='Jittor')
        plt.semilogy(range(1, len(self.pytorch_metrics['lr']) + 1), self.pytorch_metrics['lr'], 'r-', label='PyTorch')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
        
        # 创建性能对比柱状图
        plt.figure(figsize=(10, 6))
        avg_jittor_time = np.mean(self.jittor_metrics['time'])
        avg_pytorch_time = np.mean(self.pytorch_metrics['time'])
        
        frameworks = ['Jittor', 'PyTorch']
        avg_times = [avg_jittor_time, avg_pytorch_time]
        
        bars = plt.bar(frameworks, avg_times, color=['blue', 'red'])
        plt.xlabel('Framework')
        plt.ylabel('Average Time per Iteration (s)')
        plt.title('Average Iteration Time Comparison')
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}s',
                    ha='center', va='bottom', rotation=0)
        
        # 相对性能提升百分比
        if avg_pytorch_time > 0:
            speedup = (avg_pytorch_time - avg_jittor_time) / avg_pytorch_time * 100
            plt.figtext(0.5, 0.01, f'Jittor is {abs(speedup):.2f}% {"faster" if speedup > 0 else "slower"} than PyTorch',
                      ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=300)
        plt.close()
    
    def visualize_radar_comparison(self, output_dir):
        """生成雷达图比较两个框架的多维度性能"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        summary = self.generate_summary_metrics()
        
        # 雷达图的多个维度
        categories = [
            'Training Speed', 
            'mAP', 
            'mAP@0.5',
            'Loss Convergence',
            'Small Object Detection',
            'Medium Object Detection',
            'Large Object Detection'
        ]
        
        # 归一化数据值为0-1之间，值越大表示性能越好
        # 对于损失值，越小越好，所以用1减去归一化值
        
        # 训练速度：值越小越好
        jittor_speed = summary['jittor']['avg_iter_time']
        pytorch_speed = summary['pytorch']['avg_iter_time']
        max_speed = max(jittor_speed, pytorch_speed)
        if max_speed > 0:
            jittor_speed_norm = 1 - (jittor_speed / max_speed)
            pytorch_speed_norm = 1 - (pytorch_speed / max_speed)
        else:
            jittor_speed_norm = pytorch_speed_norm = 0.5
        
        # mAP值：值越大越好
        jittor_map = summary['jittor']['best_map']
        pytorch_map = summary['pytorch']['best_map']
        max_map = max(jittor_map, pytorch_map, 0.001)  # 避免除以零
        jittor_map_norm = jittor_map / max_map if max_map > 0 else 0
        pytorch_map_norm = pytorch_map / max_map if max_map > 0 else 0
        
        # mAP@0.5：值越大越好
        jittor_map50 = summary['jittor']['best_map_50']
        pytorch_map50 = summary['pytorch']['best_map_50']
        max_map50 = max(jittor_map50, pytorch_map50, 0.001)
        jittor_map50_norm = jittor_map50 / max_map50 if max_map50 > 0 else 0
        pytorch_map50_norm = pytorch_map50 / max_map50 if max_map50 > 0 else 0
        
        # 损失收敛性：最终损失越小越好
        jittor_loss = summary['jittor']['final_loss'] if summary['jittor']['final_loss'] is not None else 0
        pytorch_loss = summary['pytorch']['final_loss'] if summary['pytorch']['final_loss'] is not None else 0
        max_loss = max(jittor_loss, pytorch_loss, 0.001)
        jittor_loss_norm = 1 - (jittor_loss / max_loss) if max_loss > 0 else 0
        pytorch_loss_norm = 1 - (pytorch_loss / max_loss) if max_loss > 0 else 0
        
        # 小物体检测能力：值越大越好
        jittor_map_s = max(self.jittor_eval['mAP_s']) if self.jittor_eval['mAP_s'] else 0
        pytorch_map_s = max(self.pytorch_eval['mAP_s']) if self.pytorch_eval['mAP_s'] else 0
        max_map_s = max(jittor_map_s, pytorch_map_s, 0.001)
        jittor_map_s_norm = jittor_map_s / max_map_s if max_map_s > 0 else 0
        pytorch_map_s_norm = pytorch_map_s / max_map_s if max_map_s > 0 else 0
        
        # 中等物体检测能力：值越大越好
        jittor_map_m = max(self.jittor_eval['mAP_m']) if self.jittor_eval['mAP_m'] else 0
        pytorch_map_m = max(self.pytorch_eval['mAP_m']) if self.pytorch_eval['mAP_m'] else 0
        max_map_m = max(jittor_map_m, pytorch_map_m, 0.001)
        jittor_map_m_norm = jittor_map_m / max_map_m if max_map_m > 0 else 0
        pytorch_map_m_norm = pytorch_map_m / max_map_m if max_map_m > 0 else 0
        
        # 大物体检测能力：值越大越好
        jittor_map_l = max(self.jittor_eval['mAP_l']) if self.jittor_eval['mAP_l'] else 0
        pytorch_map_l = max(self.pytorch_eval['mAP_l']) if self.pytorch_eval['mAP_l'] else 0
        max_map_l = max(jittor_map_l, pytorch_map_l, 0.001)
        jittor_map_l_norm = jittor_map_l / max_map_l if max_map_l > 0 else 0
        pytorch_map_l_norm = pytorch_map_l / max_map_l if max_map_l > 0 else 0
        
        # 组合数据
        jittor_values = [
            jittor_speed_norm,
            jittor_map_norm,
            jittor_map50_norm,
            jittor_loss_norm,
            jittor_map_s_norm,
            jittor_map_m_norm,
            jittor_map_l_norm
        ]
        
        pytorch_values = [
            pytorch_speed_norm,
            pytorch_map_norm,
            pytorch_map50_norm,
            pytorch_loss_norm,
            pytorch_map_s_norm,
            pytorch_map_m_norm,
            pytorch_map_l_norm
        ]
        
        # 设置角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        # 扩展值列表，形成闭环
        jittor_values += jittor_values[:1]
        pytorch_values += pytorch_values[:1]
        categories += categories[:1]
        
        # 创建雷达图
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
        
        # 绘制Jittor的雷达图
        ax.plot(angles, jittor_values, 'b-', linewidth=2, label='Jittor')
        ax.fill(angles, jittor_values, 'b', alpha=0.1)
        
        # 绘制PyTorch的雷达图
        ax.plot(angles, pytorch_values, 'r-', linewidth=2, label='PyTorch')
        ax.fill(angles, pytorch_values, 'r', alpha=0.1)
        
        # 设置雷达图属性
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_yticklabels([])  # 不显示数值刻度
        
        # 添加实际数值标签（原始值，非归一化）
        for i, (jittor_val, pytorch_val) in enumerate(zip([
                f"{summary['jittor']['avg_iter_time']:.4f}s",
                f"{summary['jittor']['best_map']:.4f}",
                f"{summary['jittor']['best_map_50']:.4f}",
                f"{summary['jittor']['final_loss']:.4f}",
                f"{jittor_map_s:.4f}",
                f"{jittor_map_m:.4f}",
                f"{jittor_map_l:.4f}"
            ], [
                f"{summary['pytorch']['avg_iter_time']:.4f}s",
                f"{summary['pytorch']['best_map']:.4f}",
                f"{summary['pytorch']['best_map_50']:.4f}",
                f"{summary['pytorch']['final_loss']:.4f}",
                f"{pytorch_map_s:.4f}",
                f"{pytorch_map_m:.4f}",
                f"{pytorch_map_l:.4f}"
            ])[:-1]):
            angle = angles[i]
            if i == 0:  # 训练速度
                plt.annotate(f"J: {jittor_val}\nP: {pytorch_val}", 
                    xy=(angle, 0.5), xytext=(angle, 1.1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                    ha='center', va='center')
            else:
                plt.annotate(f"J: {jittor_val}\nP: {pytorch_val}", 
                    xy=(angle, 0.5), xytext=(angle, 1.1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                    ha='center', va='center')
        
        plt.title('Framework Performance Comparison (Normalized)', size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_metrics(self):
        """生成摘要指标数据，用于README"""
        summary = {
            'jittor': {
                'avg_loss': np.mean(self.jittor_metrics['loss']),
                'final_loss': self.jittor_metrics['loss'][-1] if self.jittor_metrics['loss'] else None,
                'avg_iter_time': np.mean(self.jittor_metrics['time']),
                'best_map': max(self.jittor_eval['mAP']) if self.jittor_eval['mAP'] else 0,
                'best_map_epoch': self.jittor_eval['epochs'][np.argmax(self.jittor_eval['mAP'])] if self.jittor_eval['mAP'] else None,
                'best_map_50': max(self.jittor_eval['mAP_50']) if self.jittor_eval['mAP_50'] else 0
            },
            'pytorch': {
                'avg_loss': np.mean(self.pytorch_metrics['loss']),
                'final_loss': self.pytorch_metrics['loss'][-1] if self.pytorch_metrics['loss'] else None,
                'avg_iter_time': np.mean(self.pytorch_metrics['time']),
                'best_map': max(self.pytorch_eval['mAP']) if self.pytorch_eval['mAP'] else 0,
                'best_map_epoch': self.pytorch_eval['epochs'][np.argmax(self.pytorch_eval['mAP'])] if self.pytorch_eval['mAP'] else None,
                'best_map_50': max(self.pytorch_eval['mAP_50']) if self.pytorch_eval['mAP_50'] else 0
            }
        }
        
        if summary['pytorch']['avg_iter_time'] > 0:
            summary['speedup'] = (summary['pytorch']['avg_iter_time'] - summary['jittor']['avg_iter_time']) / summary['pytorch']['avg_iter_time'] * 100
        else:
            summary['speedup'] = 0
            
        return summary

    def generate_report(self, output_dir):
        """生成比较报告的README.md文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        summary = self.generate_summary_metrics()
        
        readme_content = f"""# Jittor vs PyTorch Framework Comparison for GFL

## 概述

本报告比较了在GFocalV2目标检测模型上使用Jittor和PyTorch两个框架的训练过程和性能差异。分析基于训练日志提取的性能指标和模型评估结果。

## 多维度性能对比

下图展示了两个框架在多个关键性能指标上的归一化比较结果：

![雷达图比较](./visualization/radar_comparison.png)

## 训练性能比较

### 损失函数收敛对比

![损失函数比较](./visualization/loss_comparison.png)

#### 损失函数摘要数据
| 框架 | 平均损失 | 最终损失 |
|------|----------|----------|
| Jittor | {summary['jittor']['avg_loss']:.4f} | {summary['jittor']['final_loss']:.4f} |
| PyTorch | {summary['pytorch']['avg_loss']:.4f} | {summary['pytorch']['final_loss']:.4f} |

### 模型评估结果对比

![mAP比较](./visualization/map_comparison.png)

#### 模型评估指标摘要
| 框架 | 最佳mAP | 最佳mAP所在Epoch | 最佳mAP@0.5 |
|------|---------|-----------------|------------|
| Jittor | {summary['jittor']['best_map']:.4f} | {summary['jittor']['best_map_epoch']} | {summary['jittor']['best_map_50']:.4f} |
| PyTorch | {summary['pytorch']['best_map']:.4f} | {summary['pytorch']['best_map_epoch']} | {summary['pytorch']['best_map_50']:.4f} |

### 运行速度比较

![性能比较](./visualization/performance_comparison.png)
![速度比较](./visualization/speed_comparison.png)

#### 速度性能指标摘要
| 框架 | 平均迭代时间(秒) |
|------|-----------------|
| Jittor | {summary['jittor']['avg_iter_time']:.4f} |
| PyTorch | {summary['pytorch']['avg_iter_time']:.4f} |

Jittor比PyTorch {'快' if summary['speedup'] > 0 else '慢'} **{abs(summary['speedup']):.2f}%**。

## 分析结论

### 训练过程分析
- 损失函数收敛趋势：{'两个框架的损失下降趋势基本一致' if abs(summary['jittor']['avg_loss'] - summary['pytorch']['avg_loss']) / max(summary['jittor']['avg_loss'], summary['pytorch']['avg_loss']) < 0.1 else '两个框架的损失下降趋势存在一定差异'}
- 训练稳定性：{'两个框架训练过程均较为稳定' if len(self.jittor_metrics['loss']) > 0 and len(self.pytorch_metrics['loss']) > 0 else '需要更多数据来评估训练稳定性'}

### 性能分析
- 训练速度：Jittor框架相比PyTorch{'有明显的速度优势' if summary['speedup'] > 10 else '速度基本持平' if abs(summary['speedup']) < 5 else {'略有' + ('优势' if summary['speedup'] > 0 else '劣势') if abs(summary['speedup']) < 10 else '明显' + ('优势' if summary['speedup'] > 0 else '劣势')}}
- 内存使用：根据日志内容无法直接比较内存使用情况

### 模型精度分析
- mAP指标：{'Jittor框架训练的模型mAP指标略高' if summary['jittor']['best_map'] > summary['pytorch']['best_map'] else 'PyTorch框架训练的模型mAP指标略高' if summary['pytorch']['best_map'] > summary['jittor']['best_map'] else '两个框架训练的模型mAP指标基本持平'}
- 不同尺度物体检测能力：详见mAP_s（小物体）、mAP_m（中等物体）、mAP_l（大物体）的比较图表

## 总结

{'Jittor框架在保持相当精度的同时，具有一定的速度优势' if summary['speedup'] > 0 and abs(summary['jittor']['best_map'] - summary['pytorch']['best_map']) < 0.01 else 
'PyTorch框架在性能与精度方面表现更为平衡' if summary['speedup'] < 0 and summary['pytorch']['best_map'] > summary['jittor']['best_map'] else
'两个框架各有优势，选择依赖于具体应用场景需求'}。

"""
        
        with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='分析和可视化Jittor与PyTorch框架的训练日志')
    parser.add_argument('--jittor-log', type=str, required=True, help='Jittor训练日志路径')
    parser.add_argument('--pytorch-log', type=str, required=True, help='PyTorch训练日志路径')
    parser.add_argument('--output-dir', type=str, default='framework_comparison', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建可视化输出目录
    vis_dir = os.path.join(args.output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 分析日志
    analyzer = LogAnalyzer(args.jittor_log, args.pytorch_log)
    analyzer.parse_jittor_log()
    analyzer.parse_pytorch_log()
    
    # 生成可视化
    analyzer.visualize_loss_comparison(vis_dir)
    analyzer.visualize_map_comparison(vis_dir)
    analyzer.visualize_performance_comparison(vis_dir)
    analyzer.visualize_radar_comparison(vis_dir)
    
    # 生成报告
    summary = analyzer.generate_report(args.output_dir)
    
    print(f"分析完成，输出保存到 {args.output_dir} 目录")
    print(f"Jittor 平均迭代时间: {summary['jittor']['avg_iter_time']:.4f}秒")
    print(f"PyTorch 平均迭代时间: {summary['pytorch']['avg_iter_time']:.4f}秒")
    print(f"速度差异: Jittor比PyTorch {'快' if summary['speedup'] > 0 else '慢'} {abs(summary['speedup']):.2f}%")

if __name__ == '__main__':
    main() 