from __future__ import print_function

import argparse
import pprint
import mxnet as mx

from ..config import config, default, generate_config
from ..symbol import *
from ..dataset import *
from ..core.loader import TestLoader, SequentialLoader
from ..core.tester import Predictor, pred_eval_mask, pred_demo_mask
from ..utils.load_model import load_param
import pdb


def demo_maskrcnn(network, dataset, image_set, root_path, dataset_path, result_path,
                  ctx, prefix, epoch, vis, shuffle, has_rpn, proposal, thresh):
    # set config
    if has_rpn:
        config.TEST.HAS_RPN = True

    # print config
    pprint.pprint(config)

    # load symbol and testing data
    if has_rpn:
        if cfg.MASKFCN.ON:
            sym = \
                eval('get_' + network + '_maskfcn_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
        else:
            sym = eval('get_' + network + '_mask_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
        imdb = eval(dataset)(image_set, root_path, dataset_path)
        roidb = imdb.gt_roidb()
    else:
        raise NotImplementedError

    # (possibly) group the roidb by aspect
    horizontal_inds, vertical_inds = [], []
    for ind, roirec in enumerate(roidb):
        if roirec['width'] > roirec['height']:
            horizontal_inds.append(ind)
        else:
            vertical_inds.append(ind)

    aspect_group = True if len(horizontal_inds) > 0 and len(vertical_inds) > 0 else False
    print("aspect_group={}".format(aspect_group))

    if aspect_group:
        horizontal_roidb = [roidb[ind] for ind in horizontal_inds]
        vertical_roidb = [roidb[ind] for ind in vertical_inds]
        roidb = horizontal_roidb + vertical_roidb
        reorder_inds = horizontal_inds + vertical_inds
        imdb.reorder(reorder_inds)
        print("horizontal roidb length: %d" % len(horizontal_roidb))
        print("vertical roidb length: %d" % len(vertical_roidb))
        l1 = TestLoader(horizontal_roidb, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)
        l2 = TestLoader(vertical_roidb, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)
        test_data = SequentialLoader(iters=[l1, l2])
    else:
        test_data = TestLoader(roidb, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=False, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

    # check parameters
    for k in sym.list_arguments():
        if k in data_shape_dict or 'label' in k:
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

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = None
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    if not has_rpn:
        max_data_shape.append(('rois', (1, config.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    pred_demo_mask(predictor, test_data, imdb, roidb, result_path, vis=vis, thresh=thresh)
