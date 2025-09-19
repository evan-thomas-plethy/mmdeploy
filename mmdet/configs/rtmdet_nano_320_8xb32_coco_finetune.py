_base_ = 'rtmdet_l_8xb32-300e_coco.py'

# Load pretrained checkpoint for fine-tuning
load_from = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'

input_shape = 320

# Optimized training configuration for 10k images fine-tuning
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # Reduced from 300 for fine-tuning
    val_interval=1,  # Validation interval
    dynamic_intervals=[(40, 1)])  # Switch to stage2 at epoch 40

# Optimized learning rate for fine-tuning
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,  # Lower start factor for fine-tuning
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,  # Lower minimum LR for fine-tuning
        begin=10,  # Start cosine annealing earlier
        end=50,
        T_max=40,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Optimized optimizer for fine-tuning
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.001,  # Lower learning rate for fine-tuning
        weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.25,
        use_depthwise=True,
    ),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True,
    ),
    bbox_head=dict(
        in_channels=64,
        feat_channels=64,
        share_conv=False,
        exp_on_reg=False,
        use_depthwise=True,
        num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(input_shape, input_shape),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(input_shape * 2, input_shape * 2),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_shape, input_shape)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(input_shape, input_shape),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_shape, input_shape)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_shape, input_shape), keep_ratio=True),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Optimized dataloader for 10k images
train_dataloader = dict(
    batch_size=32,  # Increased batch size for better stability
    num_workers=8,  # More workers for faster data loading
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline, metainfo=dict(classes=('person', ))))

val_dataloader = dict(
    batch_size=16,  # Smaller batch size for validation
    num_workers=4,
    persistent_workers=True,
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=('person', ))))
test_dataloader = val_dataloader

# Optimized hooks for fine-tuning
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,  # Lower momentum for fine-tuning
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=40,  # Switch to stage2 earlier for fine-tuning
        switch_pipeline=train_pipeline_stage2)
]

# Checkpoint saving configuration
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save checkpoint every 5 epochs
        max_keep_ckpts=3,  # Keep only 3 best checkpoints
        save_best='auto'))  # Save best model automatically