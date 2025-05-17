from solver.ddp_mix_solver import DDPMixSolver
import torch.distributed as dist

# python -m torch.distributed.launch --nproc_per_node=4 main.py

if __name__ == '__main__':
    # 初始化分布式进程组
    # dist.init_process_group(backend='nccl', init_method='env://')
    processor = DDPMixSolver(cfg_path="config/gfocal.yaml")
    processor.run()
    # 程序结束前销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()
