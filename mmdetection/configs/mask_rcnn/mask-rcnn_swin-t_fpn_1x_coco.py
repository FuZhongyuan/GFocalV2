_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1280, 800), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        gap_before_final_norm=False),
    neck=dict(
        type='FPN',
        in_channels= [96, 192, 384, 768],
        out_channels=256,
        num_outs=5)
    )

# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='Conv2former',
#         arch='t',
#         out_indices=(0, 1, 2, 3)),
#     neck=dict(
#         in_channels=[72, 144, 288, 576])
#     )

# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.5,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         with_cp=False,
#         convert_weights=True,),
#     neck=dict(in_channels=[96, 192, 384, 768])
#     )