from solver.ddp_mix_solver import DDPMixSolver

if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="config/gfocal.yaml")
    processor.run() 