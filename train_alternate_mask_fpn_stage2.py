import argparse
import logging

import mxnet as mx
import sys
sys.path.append('/mnt/truenas/scratch/siyu/keypoint_maskrcnn')
import rcnn
from rcnn.config import config, default, generate_config
from rcnn.tools.train_maskrcnn import train_maskrcnn


def alternate_train(args, ctx, pretrained, epoch,
                    rpn_epoch, rpn_lr, rpn_lr_step,
                    rcnn_epoch, rcnn_lr, rcnn_lr_step):
    # set up logger
    # logging.basicConfig(filename="mask_rcnn_alternate_train_%d.log" % int(time.time()))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    # basic config
    begin_epoch = 0
    config.TRAIN.BG_THRESH_LO = 0.0

    # model path
    model_path = args.prefix
    args.resume = True
    logging.info('########## TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    args.no_flip = True
    train_maskrcnn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
                   args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
                   ctx, pretrained, epoch, model_path + '/rcnn1', begin_epoch, rcnn_epoch,
                   train_shared=False, lr=rcnn_lr, lr_step=rcnn_lr_step, proposal='rpn', maskrcnn_stage='rcnn1')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    # rcnn.config.generate_config('resnet_fpn', 'coco')
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--no_flip', help='disable flip images', action='store_true')
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    parser.add_argument('--resume', help='continue training', action='store_true')
    # alternate
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--rpn_epoch', help='end epoch of rpn training', default=default.rpn_epoch, type=int)
    parser.add_argument('--rpn_lr', help='base learning rate', default=default.rpn_lr, type=float)
    parser.add_argument('--rpn_lr_step', help='learning rate steps (in epoch)', default=default.rpn_lr_step, type=str)
    parser.add_argument('--rcnn_epoch', help='end epoch of rcnn training', default=default.rcnn_epoch, type=int)
    parser.add_argument('--rcnn_lr', help='base learning rate', default=default.rcnn_lr, type=float)
    parser.add_argument('--rcnn_lr_step', help='learning rate steps (in epoch)', default=default.rcnn_lr_step, type=str)
    parser.add_argument('--prefix', help='new model prefix', default=default.alternate_prefix, type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # args.network = 'resnet_fpn'
    # args.dataset = 'coco'
    # args.image_set = 'minitrain2017'
    # args.root_path =  'model/res50-fpn/coco/debug/'
    # args.pretrained = 'model/resnet-50'
    # args.prefix = 'model/res50-fpn/coco/debug/'
    # args.pretrained_epoch = 0
    # args.gpus = '0'
    print 'Called with argument:', args
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    alternate_train(args, ctx, args.pretrained, args.pretrained_epoch,
                    args.rpn_epoch, args.rpn_lr, args.rpn_lr_step,
                    args.rcnn_epoch, args.rcnn_lr, args.rcnn_lr_step)


if __name__ == '__main__':
    # import os
    # os.chdir('/mnt/truenas/scratch/siyu/keypoint_maskrcnn')
    main()
