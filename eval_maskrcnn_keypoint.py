from __future__ import print_function
import argparse
import mxnet as mx
import os
from rcnn.config import config, default, generate_config
from rcnn.tools.test_maskrcnn_keypoint import test_maskrcnn_keypoint


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    #generate_config(args.network, args.dataset)
    generate_config('resnet_fpn', 'coco')

    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--result_path', help='result path', type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default='0', type=str)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly', action='store_true')
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.network = 'resnet_fpn'
    args.dataset = 'coco'
    args.image_set = 'personsample2017'
    args.prefix = 'model/res50-fpn/coco/personsample/final'
    args.result_path = 'result/keypoint_json/personsample'
    args.has_rpn = True
    args.epoch = 0
    args.gpus = '0'
    ctx = [mx.gpu(int(gpu)) for gpu in args.gpu.split(',')]
    print(args)
    test_maskrcnn_keypoint(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path, args.result_path,
                  ctx, args.prefix, args.epoch, args.vis, args.shuffle, args.has_rpn, args.proposal, args.thresh)


if __name__ == '__main__':
    os.chdir('/mnt/truenas/scratch/siyu/keypoint_maskrcnn')
    main()
