_base_ = "/scratch/jp4906/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"

model = dict(
    roi_head=dict(
       bbox_head=dict(num_classes=12)))

dataset_type = 'COCODataset'
classes = ('Baton','Bullet','Gun','Hammer','HandCuffs','Knife','Lighter','Pliers','Powerbank','Scissors','Sprayer','Wrench')
data_root = '/scratch/jp4906/pidray/'
data = dict(
    samples_per_gpu=16,
    train=dict(
        img_prefix=data_root + 'train/',
        classes=classes,
        ann_file=data_root+'annotations/xray_train.json'),
    val=dict(
        img_prefix=data_root+'easy/',
        classes=classes,
        ann_file=data_root+'annotations/xray_test_easy.json'),
    test=dict(
        img_prefix=data_root+'easy/',
        classes=classes,
        ann_file=data_root+'annotations/xray_test_easy.json',
        samples_per_gpu=16))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=5)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
