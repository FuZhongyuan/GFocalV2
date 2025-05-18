import jittor as jt
import yaml
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入求解器
from solver.ddp_mix_solver import DDPMixSolver

if __name__ == '__main__':
    # 设置Jittor后端配置
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    
    # 初始化求解器
    processor = DDPMixSolver(cfg_path="config/gfocal.yaml")
    processor.run() 