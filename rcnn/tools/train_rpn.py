from __future__ import print_function

import argparse
import logging
import pprint

from ..config import default, generate_config
from ..symbol import *
from ..core import callback, metric
from ..core.scheduler import WarmupMultiFactorScheduler
from ..io.threaded_loader import ThreadedAnchorLoaderFPN, ThreadedAspectAnchorLoaderFPN
from ..core.module import MutableModule
from ..utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from ..utils.load_model import load_param


def train_rpn(network, dataset, image_set, root_path, dataset_path,
              frequent, kvstore, work_load_list, no_flip, no_shuffle, resume,
              ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              train_shared, lr, lr_step):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load symbol
    sym = eval('get_' + network + '_rpn')(num_anchors=config.NUM_ANCHORS)
    feat_sym = []
    for stride in config.RPN_FEAT_STRIDE:
        feat_sym.append(sym.get_internals()['rpn_cls_score_stride%s_output' % stride])

    # setup multi-gpu
    batch_size = len(ctx)
    config.TRAIN.BATCH_IMAGES = 1
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in image_set.split('+')]
    roidbs = [load_gt_roidb(dataset, image_set, root_path, dataset_path, flip=not no_flip, load_memory=config.MEMORY)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb)

    # load training data
    AnchorLoader = ThreadedAspectAnchorLoaderFPN if config.TRAIN.ASPECT_GROUPING else ThreadedAnchorLoaderFPN
    print(AnchorLoader)
    train_data = AnchorLoader(mx.sym.Group(feat_sym), roidb, batch_size=input_batch_size, shuffle=not no_shuffle,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS, short=config.SCALES[0][0],
                              long=config.SCALES[0][1], allowed_border=9999, num_thread=16)

    # infer max shape
    max_data_shape, max_label_shape = train_data.provide_data, train_data.provide_label

    # infer shape
    data_shape_dict = {i[0]: i[1] for i in (train_data.provide_data + train_data.provide_label)}
    print(data_shape_dict)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = zip(sym.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print('output shape')
    pprint.pprint(out_shape_dict)

    # load and initialize params
    if resume:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)

        # tweak the bn as kaiming's
        mergable_bn_names = sorted([k[:-5] for k in arg_params.keys() if 'bn' in k and 'bn_data' not in k and 'beta' in k])
        pprint.pprint(mergable_bn_names)
        for bn_name in mergable_bn_names:
            print("merging: %s" % bn_name)
            beta, gamma = arg_params[bn_name + '_beta'], arg_params[bn_name + '_gamma']
            mmean, mvar = aux_params[bn_name + '_moving_mean'], aux_params[bn_name + '_moving_var']
            inv_sqrt_mvar = 1 / mx.nd.sqrt(mvar + 1e-10)
            beta = beta - mmean * gamma * inv_sqrt_mvar
            gamma = gamma * inv_sqrt_mvar
            mmean[:] = 0
            mvar[:] = 1
            arg_params[bn_name + '_beta'], arg_params[bn_name + '_gamma'] = beta, gamma
            aux_params[bn_name + '_moving_mean'], aux_params[bn_name + '_moving_var'] = mmean, mvar

        msra = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        normal001 = mx.init.Normal(sigma=0.01)
        for k in sym.list_arguments():
            if k in data_shape_dict:
                continue
            if k not in arg_params:
                arg_params[k] = mx.nd.empty(shape=arg_shape_dict[k])
                if k.endswith('bias'):
                    print('init %s with zero' % k)
                    arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
                elif k in ['rpn_conv_weight', 'rpn_conv_cls_weight', 'rpn_conv_bbox_weight']:
                    print('init %s with normal(0.01)' % k)
                    normal001(k, arg_params[k])
                else:
                    print('init %s with msra' % k)
                    msra(k, arg_params[k])

        for k in sym.list_auxiliary_states():
            if k not in aux_params:
                print('init', k)
                aux_params[k] = mx.nd.empty(shape=aux_shape_dict[k])
                msra(k, aux_params[k])

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(
                arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(
                aux_params[k].shape)

    # create solver
    data_names = train_data.data_name
    label_names = train_data.label_name
    if train_shared:
        fixed_param_prefix = config.FIXED_PARAMS_SHARED
    else:
        fixed_param_prefix = config.FIXED_PARAMS

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    eval_metric = metric.RPNAccMetric()
    cls_metric = metric.RPNLogLossMetric()
    bbox_metric = metric.RPNRegLossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()

    simple_metric = True
    if simple_metric:
        for child_metric in [eval_metric, cls_metric, bbox_metric]:
            eval_metrics.add(child_metric)
    else:
        for child_metric in [eval_metric, cls_metric, bbox_metric]:
            eval_metrics.add(child_metric)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    # decide learning rate
    base_lr = lr / 8 * len(ctx)
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, warmup=True, warmup_type="gradual", warmup_step=500)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0001,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size)}
                        #'clip_gradient': 5}
    train_data = mx.io.PrefetchingIter(train_data)
    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kvstore, optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)

    del mod


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Region Proposal Network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
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
    # rpn
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--prefix', help='new model prefix', default=default.rpn_prefix, type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=default.rpn_epoch, type=int)
    parser.add_argument('--lr', help='base learning rate', default=default.rpn_lr, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.rpn_lr_step, type=str)
    parser.add_argument('--train_shared', help='second round train shared params', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Called with argument: {}'.format(args))
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_rpn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
              args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
              ctx, args.pretrained, args.pretrained_epoch, args.prefix, args.begin_epoch, args.end_epoch,
              train_shared=args.train_shared, lr=args.lr, lr_step=args.lr_step)


if __name__ == '__main__':
    main()
