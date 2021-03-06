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

    pprint.pprint(config)

    # set up maskrcnn
    cfg.MASKFCN.ON = False #####??

    # set up logger
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    logger = logging.getLogger()

    # load symbol
    if config.KEYPOINT.USE_HEATMAP:
        if config.KEYPOINT.USE_L2:
            sym = get_resnet_fpn_maskrcnn_keypoint_heatmap_L2(num_classes=config.NUM_CLASSES)
        else:
            sym = get_resnet_fpn_maskrcnn_keypoint_heatmap(num_classes=config.NUM_CLASSES)
    else:
        sym = eval('get_' + network + '_maskrcnn')(num_classes=config.NUM_CLASSES) # get_resnet_fpn_maskrcnn

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size # 1 * num of gpus
    print('input_batch_size:', input_batch_size)

    roidb_file = root_path + '/cache/' + dataset + '_roidb_with_mask.pkl'
    mean_file = root_path + '/cache/' + dataset + '_roidb_mean.pkl'
    std_file = root_path + '/cache/' + dataset + '_roidb_std.pkl'

    # mask rcnn _stage = rcnn1
    if maskrcnn_stage is not None:
        roidb_file = root_path + '/cache/' + dataset + '_roidb_with_mask_' + maskrcnn_stage + '.pkl'
        mean_file = root_path + '/cache/' + dataset + '_roidb_mean_' + maskrcnn_stage + '.pkl'
        std_file = root_path + '/cache/' + dataset + '_roidb_std_' + maskrcnn_stage + '.pkl'

    #### removed False
    if osp.exists(roidb_file) and osp.exists(mean_file) and osp.exists(std_file):
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

        # load proposal from rpn generated rois, also append ground truth boxes as rois for training.
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
    print("Use ASPECT_GROUPING:", config.TRAIN.ASPECT_GROUPING)
    MaskROIIter = ThreadedAspectMaskROIIter if config.TRAIN.ASPECT_GROUPING else ThreadedMaskROIIter
    print("MaskROIIter:", MaskROIIter)
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
    if config.KEYPOINT.USE_HEATMAP:
        max_label_shape.append(('keypoint_target', (input_batch_size, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), 17, config.KEYPOINT.MAPSIZE, config.KEYPOINT.MAPSIZE)))
        if config.KEYPOINT.USE_L2:
            max_label_shape.append(('keypoint_weight', (input_batch_size, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), 17, config.KEYPOINT.MAPSIZE, config.KEYPOINT.MAPSIZE)))
    else:
        max_label_shape.append(('keypoint_target', (input_batch_size, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), 17, 1)))

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    print('input shape:')
    pprint.pprint(data_shape_dict)
    if config.KEYPOINT.USE_HEATMAP:
        data_shape_dict['keypoint_target'] = (1, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), 17, config.KEYPOINT.MAPSIZE, config.KEYPOINT.MAPSIZE)
        if config.KEYPOINT.USE_L2:
            data_shape_dict['keypoint_weight'] = (1, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), 17, config.KEYPOINT.MAPSIZE, config.KEYPOINT.MAPSIZE)
    else:
        data_shape_dict['keypoint_target'] = (1, int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION), 17, 1)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = zip(sym.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print('output shape:')
    pprint.pprint(out_shape_dict)

    # load or initialize params
    if resume:
        print('resume', prefix)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
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
                elif k in ['kp_deconv_1_weight']:
                    print('init %s with msra' % k)
                    msra(k, arg_params[k])
                    # print('init %s with normal(0.001)' % k)
                    # normal0001(k, arg_params[k])
                elif 'P' in k:
                    print('init %s with xavier' % k)
                    xavier(k, arg_params[k])
                elif k.endswith('weight'):
                    print('init %s with msra' % k)
                    msra(k, arg_params[k])
                else:
                    raise KeyError("unknown parameter tpye %s" % k)
            else:
                print(k, 'loaded from pretrained model')
    else:
        print('pretrained', pretrained)
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
                elif k in ['rcnn_fc_bbox_weight']:
                    print('init %s with normal(0.001)' % k)
                    normal0001(k, arg_params[k])
                elif k in ['rcnn_fc_cls_weight']:
                    print('init %s with normal(0.01)' % k)
                    normal001(k, arg_params[k])
                elif k in ['rcnn_fc6_weight', 'rcnn_fc7_weight']:
                    print('init %s with xavier' % k)
                    xavier(k, arg_params[k])
                elif k in ['kp_deconv_1_weight']:
                    print('init %s with msra' % k)
                    msra(k, arg_params[k])
                elif 'P' in k:
                    print('init %s with xavier' % k)
                    xavier(k, arg_params[k])
                elif k.endswith('weight'):
                    print('init %s with msra' % k)
                    msra(k, arg_params[k])
                elif k.startswith('kp_bn'):
                    print('init %s with normal(0.01):' % k)
                    normal001(k, arg_params[k])
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
    print('data_names', data_names)
    print('label_names', label_names)
    if train_shared:
        fixed_param_prefix = config.FIXED_PARAMS_SHARED
    else:
        fixed_param_prefix = config.FIXED_PARAMS

    # fix block 0,1 and rcnn
    # fixed_param_prefix += ['rcnn']

    # # fix_all but keypoint head
    # fixed_param_prefix = ['rcnn', 'conv0', 'stage1', 'stage2', 'stage3', 'stage4',
    #                       'P5', 'P4', 'P3', 'P2', 'gamma', 'beta']
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNRegLossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    kp_log_loss = metric.KeypointLossMetric()
    kp_dummy_log_loss = mx.metric.Loss(output_names=["kp_output_output"])
    kp_l2_loss = metric.KeypointL2Metric()

    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    if config.KEYPOINT.USE_HEATMAP:
        if config.KEYPOINT.USE_L2:
             eval_metrics.add(kp_l2_loss)
        else:
             eval_metrics.add(kp_dummy_log_loss)
    else:
        eval_metrics.add(kp_log_loss)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    # decide learning rate
    base_lr = lr / 8 * len(ctx)
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    print(lr_epoch)
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, warmup=True, warmup_type="gradual", warmup_step=1)
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


    # #####################################
    # # DEBUG USE ONLY
    # #####################################
    # from mxnet.initializer import Uniform
    # from mxnet.model import BatchEndParam
    # from mxnet.base import _as_list
    # from mxnet import metric as bMetric
    # import time
    # import cPickle
    # import os
    #
    # def myfit(mod, train_data, eval_data=None, eval_metric='acc',
    #         epoch_end_callback=None, batch_end_callback=None, kvstore='local',
    #         optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
    #         eval_end_callback=None,
    #         eval_batch_end_callback=None, initializer=Uniform(0.01),
    #         arg_params=None, aux_params=None, allow_missing=False,
    #         force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
    #         validation_metric=None, monitor=None):
    #
    #     assert num_epoch is not None, 'please specify number of epochs'
    #
    #     mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
    #               for_training=True, force_rebind=force_rebind)
    #
    #     mod.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
    #                      allow_missing=allow_missing, force_init=force_init)
    #     mod.init_optimizer(kvstore=kvstore, optimizer=optimizer,
    #                         optimizer_params=optimizer_params)
    #
    #     if not isinstance(eval_metric, bMetric.EvalMetric):
    #         eval_metric = bMetric.create(eval_metric)

    #     ################################################################################
    #     # training loop
    #     ################################################################################
    #     for epoch in range(begin_epoch, num_epoch):
    #         tic = time.time()
    #         eval_metric.reset()
    #         nbatch = 0
    #         data_iter = iter(train_data)
    #         end_of_batch = False
    #         next_data_batch = next(data_iter)
    #         cnt = 0
    #         while not end_of_batch:
    #             data_batch = next_data_batch
    #             print('----------DEBUG---------')
    #             debug_save_dir = 'debug/onesample/'
    #             cPickle.dump(data_batch.data, open(os.path.join(debug_save_dir, 'data{}.pkl'.format(cnt)), 'w'))
    #             cPickle.dump(data_batch.label, open(os.path.join(debug_save_dir, 'label{}.pkl'.format(cnt)), 'w'))
    #
    #             # print(type(data_batch.data))
    #             # print(len(data_batch.label))
    #             # print((data_batch.data[0]))
    #             # print((data_batch.data[0].shape))
    #             # print((data_batch.data[1].shape))
    #             # print((data_batch.data[2].shape))
    #             # print((data_batch.data[3].shape))
    #             # print((data_batch.data[4].shape))
    #             # cnt += 1
    #             # if cnt == 50:
    #             exit()
    #
    #             mod.forward_backward(data_batch)
    #             mod.update()
    #             try:
    #                 # pre fetch next batch
    #                 next_data_batch = next(data_iter)
    #                 mod.prepare(next_data_batch)
    #             except StopIteration:
    #                 end_of_batch = True
    #
    #             mod.update_metric(eval_metric, data_batch.label)
    #
    #             if batch_end_callback is not None:
    #                 batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
    #                                                  eval_metric=eval_metric,
    #                                                  locals=locals())
    #                 for callback in _as_list(batch_end_callback):
    #                     callback(batch_end_params)
    #             nbatch += 1
    #
    #         # one epoch of training is finished
    #         for name, val in eval_metric.get_name_value():
    #             mod.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
    #         toc = time.time()
    #         mod.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
    #
    #         # sync aux params across devices
    #         arg_params, aux_params = mod.get_params()
    #         mod.set_params(arg_params, aux_params)
    #
    #         if epoch_end_callback is not None:
    #             for callback in _as_list(epoch_end_callback):
    #                 callback(epoch, mod.symbol, arg_params, aux_params)
    #
    #         # # ----------------------------------------
    #         # # evaluation on validation set
    #         # if eval_data:
    #         #     res = self.score(eval_data, validation_metric,
    #         #                      score_end_callback=eval_end_callback,
    #         #                      batch_end_callback=eval_batch_end_callback, epoch=epoch)
    #         #     # TODO: pull this into default
    #         #     for name, val in res:
    #         #         self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
    #
    #         # end of 1 epoch, reset the data-iter for another epoch
    #         train_data.reset()
    #
    # myfit(mod, train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
    #         batch_end_callback=batch_end_callback, kvstore=kvstore,
    #         optimizer='sgd', optimizer_params=optimizer_params,
    #         arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)
