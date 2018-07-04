from __future__ import print_function

import logging
import pprint
import numpy as np
import os.path as osp

from six.moves import cPickle as pkl
from ..symbol import *
from ..core import callback, metric
from ..core.scheduler import WarmupMultiFactorScheduler
from ..io.threaded_loader import ThreadedMaskROIIter, ThreadedAspectMaskROIIter
from ..core.module import MutableModule
from ..processing.bbox_regression import add_bbox_regression_targets, add_mask_targets
from ..utils.load_data import load_proposal_roidb, merge_roidb  # , filter_roidb
from ..utils.load_model import load_param


def filter_roidb(roidb):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry['max_overlaps']
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 and len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after))

    return filtered_roidb


def train_maskrcnn(network, dataset, image_set, root_path, dataset_path,
                   frequent, kvstore, work_load_list, no_flip, no_shuffle, resume,
                   ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
                   train_shared, lr, lr_step, proposal, maskrcnn_stage=None):
    # set up maskrcnn
    cfg.MASKFCN.ON = False

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load symbol
    sym = eval('get_' + network + '_maskrcnn')(num_classes=config.NUM_CLASSES)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size
    print(input_batch_size)
    # print config
    pprint.pprint(config)

    roidb_file = root_path + '/cache/' + dataset + '_roidb_with_mask.pkl'
    mean_file = root_path + '/cache/' + dataset + '_roidb_mean.pkl'
    std_file = root_path + '/cache/' + dataset + '_roidb_std.pkl'
    if maskrcnn_stage is not None:
        roidb_file = root_path + '/cache/' + dataset + '_roidb_with_mask_' + maskrcnn_stage + '.pkl'
        mean_file = root_path + '/cache/' + dataset + '_roidb_mean_' + maskrcnn_stage + '.pkl'
        std_file = root_path + '/cache/' + dataset + '_roidb_std_' + maskrcnn_stage + '.pkl'

    if False and osp.exists(roidb_file) and osp.exists(mean_file) and osp.exists(std_file):
        print('Load ' + roidb_file)
        with open(roidb_file, 'r') as f:
            roidb = pkl.load(f)
        print('Load ' + mean_file)
        with open(mean_file, 'r') as f:
            means = pkl.load(f)
        print('Load ' + std_file)
        with open(std_file, 'r') as f:
            stds = pkl.load(f)
    else:
        # load dataset and prepare imdb for training
        image_sets = [iset for iset in image_set.split('+')]
        roidbs = [load_proposal_roidb(dataset, image_set, root_path, dataset_path, proposal=proposal, append_gt=True,
                                      flip=not no_flip, load_memory=config.MEMORY, use_mask=True)
                  for image_set in image_sets]
        roidb = merge_roidb(roidbs)
        roidb = filter_roidb(roidb)
        means, stds = add_bbox_regression_targets(roidb)
        # add_assign_targets(roidb)
        if cfg.DATASET != "coco":
            # coco rasterizes mask on the fly to avoid resize artifacts
            add_mask_targets(roidb)
        for fname, obj in zip([roidb_file, mean_file, std_file], [roidb, means, stds]):
            with open(fname, 'w') as f:
                pkl.dump(obj, f, -1)

    # load training data
    MaskROIIter = ThreadedAspectMaskROIIter if config.TRAIN.ASPECT_GROUPING else ThreadedMaskROIIter
    train_data = MaskROIIter(roidb, batch_size=input_batch_size, shuffle=not no_shuffle,
                             short=config.SCALES[0][0], long=config.SCALES[0][1], num_thread=16)

    # infer max shape
    max_data_shape = [
        ('data', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    for s in config.RCNN_FEAT_STRIDE:
        max_data_shape.append(('rois_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS, 5)))
    max_label_shape = list()
    max_label_shape.append(('label', (input_batch_size, config.TRAIN.BATCH_ROIS)))
    max_label_shape.append(('bbox_target', (input_batch_size, config.TRAIN.BATCH_ROIS, config.NUM_CLASSES * 4)))
    max_label_shape.append(('bbox_weight', (input_batch_size, config.TRAIN.BATCH_ROIS, config.NUM_CLASSES * 4)))
    max_label_shape.append(('mask_target', (input_batch_size, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), config.NUM_CLASSES, 28, 28)))
    max_label_shape.append(('mask_weight', (input_batch_size, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), config.NUM_CLASSES, 28, 28)))

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    print('input shape')
    pprint.pprint(data_shape_dict)

    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = zip(sym.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print('output shape')
    pprint.pprint(out_shape_dict)

    # load or initialize params
    if resume:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        normal0001 = mx.init.Normal(sigma=0.001)
        normal001 = mx.init.Normal(sigma=0.01)
        msra = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        xavier = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)
        for k in sym.list_arguments():
            if k in data_shape_dict:
                continue
            if k not in arg_params:
                arg_params[k] = mx.nd.empty(shape=arg_shape_dict[k])
                if k.endswith('bias'):
                    print('init %s with zero' % k)
                    arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
                elif k in ['rcnn_fc_bbox_weight', 'mask_deconv_2_weight']:
                    print('init %s with normal(0.001)' % k)
                    normal0001(k, arg_params[k])
                elif k in ['rcnn_fc_cls_weight']:
                    print('init %s with normal(0.01)' % k)
                    normal001(k, arg_params[k])
                elif k in ['rcnn_fc6_weight', 'rcnn_fc7_weight']:
                    print('init %s with xavier' % k)
                    xavier(k, arg_params[k])
                elif 'P' in k:
                    print('init %s with xavier' % k)
                    xavier(k, arg_params[k])
                elif k.endswith('weight'):
                    print('init %s with msra' % k)
                    msra(k, arg_params[k])
                else:
                    raise KeyError("unknown parameter tpye %s" % k)

        for k in sym.list_auxiliary_states():
            if k not in aux_params:
                print('init %s' % k)
                aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
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

    # prepare training
    # create solver
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
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
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNRegLossMetric()
    mask_acc_metric = metric.MaskAccMetric()
    mask_log_metric = metric.MaskLogLossMetric()
    mask_log_loss = mx.metric.Loss(output_names=["mask_output_output"])
    eval_metrics = mx.metric.CompositeEvalMetric()

    simple_metric = False
    if simple_metric:
        for child_metric in [eval_metric, cls_metric, bbox_metric, mask_log_loss]:
            eval_metrics.add(child_metric)
    else:
        for child_metric in [eval_metric, cls_metric, bbox_metric, mask_acc_metric, mask_log_metric]:
            eval_metrics.add(child_metric)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
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

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)
