import os.path as osp
import numpy as np
from PIL import Image

import mmcv 
from mmcv import Config

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor


# set the data root and the directory names of images and labels
data_root = '/project/peilab/dataset/SAR/'
img_dir = 'images'
ann_dir = 'labels'

classes = ('Background', 'Tool clasper', 'Tool wrist', 'Tool shaft', 'Suturing needle', 'Thread', 'Suction tool', 'Needle Holder', 'Clamps', 'Catheter')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51], [120, 240, 90], [244, 20, 57]]


@DATASETS.register_module()
class SARRARP50(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None


cfg = Config.fromfile('configs/upernet/upernet_r50_512x512_40k_voc12aug.py')

cfg.device = 'cuda'

# Because I used 1 gpu to train the model, the argument type is 'BN'. If you have multiple gpus, you can input 'SyncBN'
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# Set the number of output classes
cfg.model.decode_head.num_classes = 10
cfg.model.auxiliary_head.num_classes = 10

# Set data_type and data_root
cfg.dataset_type = 'SARRARP50'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 2

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# cfg.crop_size = (256, 256)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = '/project/peilab/dataset/SAR/splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = '/project/peilab/dataset/SAR/splits/test.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = '/project/peilab/dataset/SAR/splits/test.txt'

# load the pretrained weights of Upernet
cfg.load_from = '/home/hchener/logs/work_dirs/latest.pth'

# set the saved directory path of checkpoints and logs
cfg.work_dir = '/home/hchener/logs/work_dirs'

cfg.runner.max_iters = 100000
cfg.log_config.interval = 500
cfg.evaluation.interval = 2000
cfg.checkpoint_config.interval = 2000

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.resume_from = '/home/hchener/logs/work_dirs/iter_68000.pth'

print(f'Config:\n{cfg.pretty_text}')


datasets = [build_dataset(cfg.data.train)]

model = build_segmentor(cfg.model)

model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())