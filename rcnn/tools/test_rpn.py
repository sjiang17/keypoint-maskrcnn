import os
import sys
import cPickle
import argparse
import pprint

from pprint import pprint
from rcnn.config import config, default, generate_config
from rcnn.symbol import *
from rcnn.dataset import *
from rcnn.core.loader import TestLoader, SequentialLoader
from rcnn.core.tester import Predictor, generate_proposals
from rcnn.utils.load_model import load_param


def pprint_with_newlines(arg, section_title, num_newline=2):
    print("\n" * num_newline)
    print("%s" % section_title)
    pprint(arg)
    print("\n" * num_newline)


def test_rpn(network, dataset, image_set, root_path, dataset_path,
             ctx, prefix, epoch, vis, shuffle, thresh):
    # rpn generate proposal config
    config.TEST.HAS_RPN = True

    # print config
    pprint(config)

    # load symbol
    sym = eval('get_' + network + '_rpn_test')(num_anchors=config.NUM_ANCHORS)

    # load dataset and prepare imdb for training
    imdb = eval(dataset)(image_set, root_path, dataset_path)
    roidb = imdb.gt_roidb()

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
        l1 = TestLoader(horizontal_roidb, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)
        l2 = TestLoader(vertical_roidb, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)
        test_data = SequentialLoader(iters=[l1, l2])
    else:
        test_data = TestLoader(roidb, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)
    
    # sanity check
    _, out_shape, _ = sym.get_internals().infer_shape(**dict(test_data.provide_data))
    out_names = sym.get_internals().list_outputs()
    pprint_with_newlines(zip(out_names, out_shape), "output shape: ")

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=None)

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
    label_names = None if test_data.provide_label is None else [k[0] for k in test_data.provide_label]
    max_data_shape = [('data', (len(ctx), 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start testing
    imdb_boxes, original_boxes = generate_proposals(predictor, test_data, imdb, vis=vis, thresh=thresh)

    if aspect_group:
        # imdb_boxes = [imdb_boxes[ind] for ind in (horizontal_inds + vertical_inds)]
        # original_boxes = [original_boxes[ind] for ind in (horizontal_inds + vertical_inds)]
        reordered_imdb_boxes, reordered_original_boxes = [None] * len(imdb_boxes), [None] * len(imdb_boxes)
        for i, orig_ind in enumerate(horizontal_inds + vertical_inds):
            reordered_imdb_boxes[orig_ind] = imdb_boxes[i]
            reordered_original_boxes[orig_ind] = original_boxes[i]
        imdb_boxes, original_boxes = reordered_imdb_boxes, reordered_original_boxes

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)

    imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Region Proposal Network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rpn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rpn_epoch, type=int)
    # rpn
    parser.add_argument('--gpu', help='GPU device to test with', default='0', type=str)
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='rpn proposal threshold', default=0, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    ctx = [mx.gpu(int(gpu)) for gpu in args.gpu.split(',')]
    test_rpn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
             ctx, args.prefix, args.epoch, args.vis, args.shuffle, args.thresh)


if __name__ == '__main__':
    main()
