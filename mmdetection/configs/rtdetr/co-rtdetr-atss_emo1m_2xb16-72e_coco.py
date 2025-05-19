_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

train_batch_per_gpu = 16
val_batch_per_gpu = 8

# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 80
hidden_dim = 128

model = dict(
    type='CoDETR',
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=True,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='EMO',
        depths=[2, 2, 8, 3],
        stem_dim=24,
        embed_dims=[32, 48, 80, 168],
        exp_ratios=[2., 2.5, 3.0, 3.5],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'],
        act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5],
        dim_heads=[16, 16, 20, 21],
        window_sizes=[7, 7, 7, 7],
        attn_ss=[False, False, True, True],
        qkv_bias=True,
        attn_drop=0,
        drop=0.,
        drop_path=0.04036,
        v_group=False,
        attn_pre=True,
        pre_dim=0,
        pretrain='/home/ubuntu/fishworld/project/RT-DETR/rtdetr_pytorch/assets/emo_1m.pth'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[48, 80, 168],
        kernel_size=1,
        out_channels=hidden_dim,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=3),
    query_head=dict(
        type='CoDINOHead',
        num_query=300,
        num_classes=num_classes,
        in_channels=hidden_dim,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='CoRTDETR',
            num_classes=num_classes,
            with_coord_feat=False,
            num_co_heads=1,  # ATSS Aux Head
            num_feature_levels=3,
            encoder=dict(
                type='HybridEncoder',
                in_channels=[hidden_dim, hidden_dim, hidden_dim],
                out_channel=96,
                feat_strides=[8, 16, 32],
                # intra
                hidden_dim=hidden_dim,
                use_encoder_idx=[2],
                num_encoder_layers=1,
                nhead=8,
                dim_feedforward=512,
                dropout=0.,
                enc_act='gelu',
                pe_temperature=10000,
                # cross
                expansion=0.5,
                depth_mult=1,
                act='silu',
                # eval
                eval_size=[640, 640]),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=96,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=96,
                            num_levels=3,
                            dropout=0.0),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=96,
                        feedforward_channels=768,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=48,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='RTDETRVflLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=1,
            feat_channels=hidden_dim,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=300,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ]
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='MinIoURandomCrop'),
    # dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 480), (512, 512), (544, 544), (576, 576),
                (608, 608), (640, 640), (672, 672), (704, 704),
                (736, 736), (768, 768), (800, 800)],
        keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_per_gpu,
    num_workers=10,
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=val_batch_per_gpu,
    num_workers=5,
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001,
        betas=[0.9, 0.999],
        params=[
            dict(params='^(?=.*backbone)(?=.*norm).*$', lr=0.00001, weight_decay=0.),
            dict(params='^(?=.*backbone)(?!.*norm).*$', lr=0.00001),
            dict(params='^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', weight_decay=0.)
        ]),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 72
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,
        max_keep_ckpts=2,
        save_best='auto'
        ),
)

custom_hooks = [
    # ema
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49
    )
]

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[1000],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
