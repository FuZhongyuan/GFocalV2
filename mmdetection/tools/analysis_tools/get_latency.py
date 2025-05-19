import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import warnings
warnings.filterwarnings(action="ignore")
from torchanalyse import profiler, System, Unit
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchstat import stat
from mmcv.cnn import get_model_complexity_info
import torch
import torch.nn as nn 
import argparse

import torch
from mmengine.dist import get_world_size, init_dist
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist
from mmengine import Config, DictAction
from mmdet.registry import MODELS
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.structures import InstanceData


class ModelWrap(nn.Module):
    def __init__(self, model):
        super(ModelWrap, self).__init__()
        self.model = model
        self.data_samples = DetDataSample()
        img_meta = dict(img_shape=(640, 640),
                         pad_shape=(640, 640))
        gt_instances = InstanceData(metainfo=img_meta)
        gt_instances.bboxes = torch.rand((5, 4))
        gt_instances.labels = torch.zeros((5,), dtype=torch.int64)
        self.data_samples.gt_instances = gt_instances
        self.data_samples.set_metainfo(dict(ori_shape=(640, 640), 
                                            scale_factor=(0.5925925925925926, 0.5925925925925926),
                                            img_shape=(640, 640),
                                            batch_input_shape=(640, 640)))
        self.data_samples = [self.data_samples]

    @torch.no_grad()
    def forward(self, tensor):
            self.model.forward(tensor, self.data_samples, mode='predict')


def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing '
        'benchmark metrics')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    """main
    """
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('env_cfg', {}).get('dist_cfg', {}))
        distributed = True
        assert get_world_size(
        ) == 1, 'Inference benchmark does not allow distributed multi-GPU'

    cfg.distributed = distributed

    log_file = None
    if args.work_dir:
        log_file = os.path.join(args.work_dir, 'benchmark.log')
        mkdir_or_exist(args.work_dir)

    MMLogger.get_instance('mmdet', log_file=log_file, log_level='INFO')

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu') 
    #     if 'ema' in checkpoint:
    #         state = checkpoint['ema']['module']
    #     else:
    #         state = checkpoint['model']
    # else:
    #     raise AttributeError('only support resume to load model.state_dict by now.')

    # # NOTE load train mode state -> convert to deploy mode
    # cfg.model.load_state_dict(state)
    
    # class Model(nn.Module):
    #     def __init__(self, ) -> None:
    #         super().__init__()
    #         self.model = cfg.model.deploy()
    #         self.postprocessor = cfg.postprocessor.deploy()
    #         print(self.postprocessor.deploy_mode)
            
    #     def forward(self, images, orig_target_sizes):
    #         outputs = self.model(images)
    #         return self.postprocessor(outputs, orig_target_sizes)
    
    unit = Unit(unit_flop="GFLOP")
    system = System(
        unit,
        frequency=940,
        flops=123,
        onchip_mem_bw=900,
        pe_min_density_support=0.0001,
        accelerator_type="structured",
        model_on_chip_mem_implications=False,
        on_chip_mem_size=32,
    )
    
    model = MODELS.build(cfg.model)
    model = model.deploy()
    model_warp = ModelWrap(model)

    inputs = (torch.randn(1, 3, 640, 640), )
    op_df = profiler(model_warp, inputs, system, unit)
    # op_df.to_csv(args.config.split("/")[-1].replace(".py", ".csv"))    
    print(op_df['Flops (GFLOP)'].sum())
    print(op_df['Latency (msec)'].sum())
    
    # thop
    macs, params = profile(model_warp, inputs=inputs, )
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    
    # fvcore
    # model = model.deploy()
    # flops = FlopCountAnalysis(model_warp, inputs)
    # print("FLOPs: ", flops.total())
    print(parameter_count_table(model))

    # torchstat
    # stat(model_warp, (3, 640, 640))

    # mmcv
    flops, params = get_model_complexity_info(model_warp, (3, 640, 640), as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")


    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total:', total_num, 'Trainable:', trainable_num)
    # backbone_param = sum(p.numel() for p in model.backbone.parameters())
    # encoder_param = sum(p.numel() for p in model.encoder.parameters())
    # decoder_param = sum(p.numel() for p in model.decoder.parameters())
    # print('backbone:', backbone_param, 'encoder:', encoder_param, 'decoder:', decoder_param)


if __name__ == '__main__':
    main()
