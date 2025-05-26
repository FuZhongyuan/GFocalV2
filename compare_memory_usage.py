#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import psutil
import argparse
import json
import datetime
import threading
import signal
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import time
import json
import os
import subprocess
import threading
from datetime import datetime
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

class MemoryMonitor:
    def __init__(self, pid, framework, mode, interval=0.5, output_dir="memory_comparison_results"):
        self.pid = pid
        self.framework = framework
        self.mode = mode  # 'train' or 'test'
        self.interval = interval  # Sampling interval in seconds
        self.running = False
        self.memory_data = {
            'timestamps': [],
            'cpu_percent': [],
            'ram_usage_mb': [],
            'gpu_memory_mb': [],
        }
        self.output_dir = output_dir
        self.start_time = None
        Path(output_dir).mkdir(exist_ok=True)
    
    def _get_gpu_memory(self):
        """Get GPU memory usage in MB"""
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                             '--format=csv,noheader,nounits'], encoding='utf-8')
            for line in result.strip().split('\n'):
                if line.strip():
                    line_pid, memory = line.split(',')
                    if int(line_pid) == self.pid:
                        return float(memory.strip())
            return 0  # If process not found
        except (subprocess.SubprocessError, FileNotFoundError):
            return 0  # If nvidia-smi is not available
    
    def _monitor_process(self):
        try:
            process = psutil.Process(self.pid)
            self.start_time = time.time()
            
            while self.running:
                try:
                    current_time = time.time() - self.start_time
                    self.memory_data['timestamps'].append(current_time)
                    
                    # CPU usage
                    self.memory_data['cpu_percent'].append(process.cpu_percent())
                    
                    # Memory usage
                    memory_info = process.memory_info()
                    self.memory_data['ram_usage_mb'].append(memory_info.rss / 1024 / 1024)  # RSS memory in MB
                    
                    # GPU memory
                    gpu_memory = self._get_gpu_memory()
                    self.memory_data['gpu_memory_mb'].append(gpu_memory)
                    
                    time.sleep(self.interval)
                except psutil.NoSuchProcess:
                    print(f"Process {self.pid} no longer running")
                    self.running = False
                    break
        except Exception as e:
            print(f"Error during monitoring: {e}")
        finally:
            self._save_data()
            self._generate_plots()
    
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_process)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Starting monitoring {self.framework} framework {self.mode} process (PID: {self.pid})")
    
    def stop(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        print(f"Stopped monitoring {self.framework} framework {self.mode} process")
    
    def _save_data(self):
        """Save collected data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{self.framework}_{self.mode}_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'framework': self.framework,
                'mode': self.mode,
                'data': self.memory_data,
                'summary': {
                    'max_ram_mb': max(self.memory_data['ram_usage_mb']) if self.memory_data['ram_usage_mb'] else 0,
                    'avg_ram_mb': sum(self.memory_data['ram_usage_mb']) / len(self.memory_data['ram_usage_mb']) if self.memory_data['ram_usage_mb'] else 0,
                    'max_gpu_mb': max(self.memory_data['gpu_memory_mb']) if self.memory_data['gpu_memory_mb'] else 0,
                    'avg_gpu_mb': sum(self.memory_data['gpu_memory_mb']) / len(self.memory_data['gpu_memory_mb']) if self.memory_data['gpu_memory_mb'] else 0,
                    'duration_seconds': self.memory_data['timestamps'][-1] if self.memory_data['timestamps'] else 0,
                }
            }, f, indent=2)
        
        print(f"Data saved to {output_file}")
        return output_file
    
    def _generate_plots(self):
        """Generate memory usage plots"""
        if not self.memory_data['timestamps']:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # RAM usage plot
        ax1.plot(self.memory_data['timestamps'], self.memory_data['ram_usage_mb'])
        ax1.set_title(f'{self.framework} {self.mode} - RAM Usage', fontsize=16)  # Larger font size
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.grid(True)
        
        # GPU memory usage plot
        if any(self.memory_data['gpu_memory_mb']):  # If GPU data is available
            ax2.plot(self.memory_data['timestamps'], self.memory_data['gpu_memory_mb'])
            ax2.set_title(f'{self.framework} {self.mode} - GPU Memory Usage', fontsize=16)  # Larger font size
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No GPU data available', horizontalalignment='center', fontsize=14)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"{self.framework}_{self.mode}_{timestamp}.png")
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")


def run_command_with_monitoring(command, framework, mode, cwd=None):
    """Run command and monitor its memory usage"""
    print(f"Running {framework} {mode} command: {command}")
    
    # Start process
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    
    # Start monitoring
    monitor = MemoryMonitor(process.pid, framework, mode)
    monitor.start()
    
    # Collect output
    stdout, stderr = process.communicate()
    
    # Stop monitoring
    monitor.stop()
    
    # Process output
    log_file = os.path.join(monitor.output_dir, f"{framework}_{mode}_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"===== STDOUT =====\n{stdout}\n")
        f.write(f"===== STDERR =====\n{stderr}\n")
    
    print(f"{framework} {mode} completed with exit code: {process.returncode}")
    return process.returncode


def compare_results(output_dir="memory_comparison_results"):
    """Compare all results and generate a comprehensive report"""
    results = []
    
    # Load all JSON results
    for file in os.listdir(output_dir):
        if file.endswith('.json'):
            with open(os.path.join(output_dir, file), 'r') as f:
                data = json.load(f)
                results.append({
                    'framework': data['framework'],
                    'mode': data['mode'],
                    'max_ram_mb': data['summary']['max_ram_mb'],
                    'avg_ram_mb': data['summary']['avg_ram_mb'],
                    'max_gpu_mb': data['summary']['max_gpu_mb'],
                    'avg_gpu_mb': data['summary']['avg_gpu_mb'],
                    'duration_seconds': data['summary']['duration_seconds'],
                })
    
    if not results:
        print("No result files found")
        return
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, "memory_comparison_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"Comparison results saved to {csv_file}")
    
    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group data by framework and mode
    for i, metric in enumerate([('max_ram_mb', 'Max RAM Usage (MB)'), 
                              ('avg_ram_mb', 'Avg RAM Usage (MB)'),
                              ('max_gpu_mb', 'Max GPU Usage (MB)'), 
                              ('avg_gpu_mb', 'Avg GPU Usage (MB)')]):
        col = metric[0]
        title = metric[1]
        ax = axes[i//2, i%2]
        
        # Prepare data
        train_data = {row['framework']: row[col] for row in results if row['mode'] == 'train'}
        test_data = {row['framework']: row[col] for row in results if row['mode'] == 'test'}
        
        # Plot
        x = range(len(train_data))
        width = 0.35
        
        train_bars = ax.bar([i - width/2 for i in x], train_data.values(), width, label='Train')
        test_bars = ax.bar([i + width/2 for i in x], test_data.values(), width, label='Test')
        
        ax.set_title(title, fontsize=16)  # Larger font size
        ax.set_xticks(x)
        ax.set_xticklabels(train_data.keys())
        ax.legend()
        
        # Add value labels on top of bars
        for bars in [train_bars, test_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison_summary.png"))
    print(f"Comparison plot saved to {os.path.join(output_dir, 'memory_comparison_summary.png')}")
    
    # Print comparison report
    print("\n===== Memory Usage Comparison Report =====")
    print(df.to_string(index=False))
    print("\n")


def main():
    # 创建结果目录
    output_dir = "memory_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置命令
    jittor_train_cmd = "python tools/train.py /root/data-fs/GFocalV2/GFocalV2Jittor/configs/gfl/gfl_r50_fpn_coco_1x_enhanced.yml"
    jittor_test_cmd = "python tools/test.py /root/data-fs/GFocalV2/GFocalV2Jittor/work_dirs/20250520_110958/gfl_r50_fpn_coco_1x_enhanced.yml /root/data-fs/GFocalV2/GFocalV2Jittor/work_dirs/20250520_110958/epoch_12.pkl"
    
    pytorch_train_cmd = "python tools/train.py"  # 注意：用户提供的命令可能有误，应该是train.py而不是test.py
    pytorch_test_cmd = "python tools/test.py /root/data-fs/GFocalV2/GFocalV2Pytorch/work_dirs/20250520_111249/gfl_r50_fpn_1x_coco.py /root/data-fs/GFocalV2/GFocalV2Pytorch/work_dirs/20250520_111249/epoch_12.pth"
    
    # 设置工作目录
    jittor_dir = "/root/data-fs/GFocalV2/GFocalV2Jittor"
    pytorch_dir = "/root/data-fs/GFocalV2/GFocalV2Pytorch"
    
    # 运行测试命令（为了演示，我们只运行几秒钟）
    # 实际使用时可以移除--max-iters参数或调整为更大的值
    jittor_train_cmd += " --max-iters 20"  # 仅运行20次迭代
    pytorch_train_cmd += " --max-iters 20"  # 仅运行20次迭代
    
    jittor_test_cmd += " --eval-options 'max_det=100'"  # 限制检测数量
    pytorch_test_cmd += " --eval-options 'max_det=100'"  # 限制检测数量
    
    # 运行Jittor训练并监控
    run_command_with_monitoring(jittor_train_cmd, "Jittor", "train", jittor_dir)
    
    # 运行PyTorch训练并监控
    run_command_with_monitoring(pytorch_train_cmd, "PyTorch", "train", pytorch_dir)
    
    # 运行Jittor测试并监控
    run_command_with_monitoring(jittor_test_cmd, "Jittor", "test", jittor_dir)
    
    # 运行PyTorch测试并监控
    run_command_with_monitoring(pytorch_test_cmd, "PyTorch", "test", pytorch_dir)
    
    # 比较结果
    compare_results(output_dir)


if __name__ == "__main__":
    main() 