import numpy as np
from easydict import EasyDict as edict

config = edict()

# network related params
config.PIXEL_MEANS = np.array([0, 0, 0])  # image mean is in BGR
config.ROIALIGN = True

config.RPN_FEAT_STRIDE = [64, 32, 16, 8, 4]
config.RCNN_FEAT_STRIDE = [32, 16, 8, 4]

config.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
config.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'stage4',
                              'P5', 'P4', 'P3', 'P2', 'gamma', 'beta']

# dataset related params
config.DATASET = "coco"
config.NUM_CLASSES = 9
config.SCALES = [(1024, 2048)]  # first is scale (the shorter side); second is max size
config.ANCHOR_SCALES = (8,)
config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)
config.CLASS_ID = [0, 1]
config.SEG_CODE = 1000

config.MEMORY = False

config.TRAIN = edict()
# R-CNN and RPN
config.TRAIN.BATCH_IMAGES = 1
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = False

# scale
config.TRAIN.SCALE = True
config.TRAIN.SCALE_RANGE = (0.8, 1)

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 512

# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# RPN proposal
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000

config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = True
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.RPN_NMS_THRESH = 0.7

config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 1000

config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# RPN generate proposal (for RPN test, aka for RCNN train in alternate mode)
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 2000  # per fpn level
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000

config.TRAIN.PROPOSAL_PRE_NMS_TOP_N = 2000  # per fpn level
config.TRAIN.PROPOSAL_POST_NMS_TOP_N = 2000

config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.5

# MaskRCNN related
config.MASKRCNN = edict()
config.MASKRCNN.MASK_LOSS = 1.0
config.MASKRCNN.GAP = False

# MaskFCN related
config.MASKFCN = edict()
config.MASKFCN.MASK_LOSS = 4.0  # instance segmentation weight loss
config.MASKFCN.ON = False  # use ROICrop for mask
config.MASKFCN.GAP = config.MASKFCN.ON and True  # use global average pooling for maskfcn
config.MASKFCN.PAN = False  # enable semantic segmentation branch
config.MASKFCN.SEM_LOSS = 1.0  # semantic segmentation weight loss
config.MASKFCN.DILATED = False # use dilated conv in backbone
config.MASKFCN.CONSISTENT = config.MASKFCN.PAN and True
config.MASKFCN.ROICROP = False  # use ROICrop for rcnn
config.MASKFCN.RPN_BBOX_FOR_MASK = False

# Memonger
config.MEMONGER = False  # use memonger to reduce memory usage

# Other Evaluation
config.EVAL_DET_ONLY = False  # eval detection mAP only
config.EVAL_SEG_ONLY = config.MASKFCN.PAN and False  # eval segmentation IOU only

# default settings
default = edict()

# default network
default.network = 'resnet_fpn'
default.pretrained = 'model/resnet-50'
default.pretrained_epoch = 0
default.base_lr = 0.02
# default dataset
default.dataset = 'Cityscape'
default.image_set = 'train'
default.test_image_set = 'val'
default.root_path = 'data'
default.dataset_path = 'data/cityscape'
# default training
default.frequent = 10
default.kvstore = 'local'
# default rpn
default.rpn_prefix = 'model/rpn'
default.rpn_epoch = 3
default.rpn_lr = default.base_lr
default.rpn_lr_step = '6'
# default rcnn
default.rcnn_prefix = 'model/rcnn'
default.rcnn_epoch = 24
default.rcnn_lr = default.base_lr
default.rcnn_lr_step = '20'
# default alternate
default.alternate_prefix = 'model/alternate_coco_'

# network settings
network = edict()

network.vgg = edict()

network.resnet_fpn = edict()
network.resnet_fpn.pretrained = 'model/resnet-50'
network.resnet_fpn.pretrained_epoch = 0
network.resnet_fpn.PIXEL_MEANS = np.array([0, 0, 0])
network.resnet_fpn.RPN_FEAT_STRIDE = [64, 32, 16, 8, 4]
network.resnet_fpn.RCNN_FEAT_STRIDE = [32, 16, 8, 4]
network.resnet_fpn.RPN_MIN_SIZE = network.resnet_fpn.RPN_FEAT_STRIDE
network.resnet_fpn.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
network.resnet_fpn.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'stage4',
                                          'P5', 'P4', 'P3', 'P2', 'gamma', 'beta']

# dataset settings
dataset = edict()

dataset.Cityscape = edict()
dataset.Cityscape.image_set = 'train'
dataset.Cityscape.test_image_set = 'val'
dataset.Cityscape.root_path = 'data'
dataset.Cityscape.dataset_path = 'data/cityscape'
dataset.Cityscape.NUM_CLASSES = 9
dataset.Cityscape.SCALES = [(1024, 2048)]
dataset.Cityscape.ANCHOR_SCALES = (8,)
dataset.Cityscape.ANCHOR_RATIOS = (0.5, 1, 2)
dataset.Cityscape.NUM_ANCHORS = len(dataset.Cityscape.ANCHOR_SCALES) * len(dataset.Cityscape.ANCHOR_RATIOS)
dataset.Cityscape.CLASS_ID = [0, 24, 25, 26, 27, 28, 31, 32, 33]
dataset.Cityscape.SEG_CODE = 1000
dataset.Cityscape.ASPECT_GROUPING = False
dataset.Cityscape.SCALE = True
dataset.Cityscape.rpn_epoch = 8
dataset.Cityscape.rpn_lr_step = '6'
dataset.Cityscape.rcnn_epoch = 24
dataset.Cityscape.rcnn_lr_step = '20'

dataset.coco = edict()
dataset.coco.image_set = 'train2017'
dataset.coco.test_image_set = 'val2017'
dataset.coco.root_path = 'data'
dataset.coco.dataset_path = 'data/coco'
dataset.coco.NUM_CLASSES = 2
dataset.coco.SCALES = [(800, 1333)]
dataset.coco.ANCHOR_SCALES = (8,)
dataset.coco.ANCHOR_RATIOS = (0.5, 1, 2)
dataset.coco.NUM_ANCHORS = len(dataset.Cityscape.ANCHOR_SCALES) * len(dataset.Cityscape.ANCHOR_RATIOS)
dataset.coco.CLASS_ID = [0, 1]
dataset.coco.CLASS_NAME = ('__background__', 'person')
dataset.coco.SEG_CODE = 500
dataset.coco.BATCH_ROIS = 256
dataset.coco.ASPECT_GROUPING = True
dataset.coco.SCALE = False
dataset.coco.rpn_epoch = 6
dataset.coco.rpn_lr_step = '3,5'
dataset.coco.rcnn_epoch = 12
dataset.coco.rcnn_lr_step = '6,8'

config.KEYPOINT = edict()
config.KEYPOINT.MAPSIZE = 56
# config.KEYPOINT.fg_num = 64


def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in config['TRAIN']:
            config['TRAIN'][k] = v
        elif k in default:
            default[k] = v
