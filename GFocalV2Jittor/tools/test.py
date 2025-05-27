import argparse
import os.path as osp

from jittordet.engine import Runner, load_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint to load from')
    parser.add_argument('--work-dir', help='the dir to save logs')
    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help='disable cuda and use cpu to train net.')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    return parser.parse_args()


def trigger_visualization_hook(cfg, args):
    """根据命令行参数激活可视化钩子。"""
    if args.show or args.show_dir:
        # 创建可视化钩子配置
        vis_hook_cfg = dict(type='DetVisualizationHook')
        
        # 配置可视化参数
        vis_hook_cfg['enable'] = True
        if args.show:
            vis_hook_cfg['show'] = args.show
        if args.show_dir:
            vis_hook_cfg['show_dir'] = args.show_dir
        if args.wait_time:
            vis_hook_cfg['wait_time'] = args.wait_time
            
        # 确保hooks列表存在
        if 'hooks' not in cfg:
            cfg.hooks = []
        
        # 将可视化钩子添加到hooks列表中
        cfg.hooks.append(vis_hook_cfg)
    
    return cfg


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    cfg.load_from = args.checkpoint
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # 处理可视化参数
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    
    # set disable cuda
    cfg.disable_cuda = args.disable_cuda

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
