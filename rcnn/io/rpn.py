"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

import numpy as np
import numpy.random as npr

from ..config import config
from .image import get_image, tensor_vstack
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, nonlinear_transform

bbox_transform = nonlinear_transform


def get_rpn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped', 'boxes']
    :return: data, label, im_info
    """
    imgs, roidb = get_image(roidb, use_random_scale=False, random_scale_range=config.TRAIN.SCALE_RANGE,
                            fixed_multi_scales=config.SCALES, pixel_mean=config.PIXEL_MEANS)
    hw = config.SCALES[0] if imgs[0].shape[2] < imgs[0].shape[3] else config.SCALES[0][::-1]
    tensor_shape = (len(imgs), imgs[0].shape[1]) + hw     # NCHW
    im_array = tensor_vstack(imgs, shape=tensor_shape)
    im_info = np.stack([roirec['im_info'] for roirec in roidb], axis=0).astype(np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {}

    return data, label, im_info


def get_rpn_batch(roidb, max_shape=None):
    """
    prototype for rpn batch: data, im_info, gt_boxes
     :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
     :param max_shape:
     :return: data, dict of {'data', 'im_info'}
     'data': 3d tensor [batch_idx, width, height], some images are padded due to scale augmentation
     'im_info': 2d tensor [batch_idx, image_info], image_info = [original_width, original_height, image_scale]
     :return: label, dict of {'gt_boxes'}
     'gt_boxes': 3d tensor [batch_idx, roi_idx, 4+1], 4+1=(x1, y1, x2, y2, cls)
     """
    # resize, mean-sub then copy, from roidb
    imgs, roidb = get_image(roidb, use_random_scale=config.TRAIN.SCALE, random_scale_range=config.TRAIN.SCALE_RANGE,
                            fixed_multi_scales=config.SCALES, pixel_mean=config.PIXEL_MEANS)
    # pad image to uniform size AFTER mean sub
    im_array = tensor_vstack(imgs, shape=max_shape, out=None)
    im_info = []
    gt_boxes = []

    for i in range(len(roidb)):
        im_info.append(roidb[i]['im_info'])
        # gt boxes: (x1, y1, x2, y2, cls)
        if roidb[i]['gt_classes'].size > 0:
            gt_inds_i = np.where(roidb[i]['gt_classes'] != 0)[0]
            gt_boxes_i = np.empty((roidb[i]['boxes'].shape[0], 5), dtype="float32")
            gt_boxes_i[:, 0:4] = roidb[i]['boxes'][gt_inds_i, :]
            gt_boxes_i[:, 4] = roidb[i]['gt_classes'][gt_inds_i]
        else:
            gt_boxes_i = np.full(shape=(1, 5), fill_value=-1, dtype="float32")
        gt_boxes.append(gt_boxes_i[np.newaxis, :, :])

    im_info = np.array(im_info, dtype=np.float32)
    data = {'data': im_array,
            'im_info': im_info}
    # gt_boxes shape : [num_images, num_boxes, 5], type list
    label = {'gt_boxes': gt_boxes}

    return data, label


def assign_anchor_fpn(feat_shape, gt_boxes, im_info, feat_strides=(64, 32, 16, 8, 4),
                      scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_strides: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: tuple
    labels: of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    bbox_targets: of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    bbox_weights: mark the assigned anchors
    """

    def _unmap(data, count, inds, fill=0):
        """ unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    scales = np.array(scales, dtype=np.float32)

    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    for i in range(len(feat_strides)):
        base_anchors = generate_anchors(base_size=feat_strides[i], ratios=list(ratios), scales=scales)
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shape[i][-2:]
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)
        K = feat_height * feat_width

        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)

    # Concat anchors from each level
    anchors = np.concatenate(anchors_list)
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
    inds_inside = np.concatenate(inds_inside_list)
    total_anchors = sum(anchors_num_list)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float32, copy=False), gt_boxes.astype(np.float32, copy=False))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        # gt_max_overlaps.shape = (k, ); overlaps.shape = (n, k)
        # the following line select anchors which overlap with at least one gt_bbox with the highest IOU
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds
    # map up to original set of anchors
    # ignore anchors exceed the border tolerance
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        if gt_boxes.size > 0:
            print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # reshape
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    for i in range(len(feat_strides)):
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        label = labels[sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
        bbox_target = bbox_targets[
                      sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
        bbox_weight = bbox_weights[
                      sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]

        label = label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        label = label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height * feat_width, A * 4)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height * feat_width, A * 4)).transpose((0, 2, 1))

        label_list.append(label)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)

    label = {'label': label_concat,
             'bbox_target': bbox_target_concat,
             'bbox_weight': bbox_weight_concat}
    return label


def assign_pyramid_anchor(feat_shapes, gt_boxes, im_info, cfg, feat_strides=(64, 32, 16, 8, 4),
                          scales=(8,), ratios=(0.5, 1, 2), allowed_border=0, balance_scale_bg=False,):
    """
    assign ground truth boxes to anchor positions
    :param feat_shapes: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_strides: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :param balance_scale_bg: restrict the background samples for each pyramid level
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info
    scales = np.array(scales, dtype=np.float32)
    ratios = np.array(ratios, dtype=np.float32)
    assert(len(feat_shapes) == len(feat_strides))

    fpn_args = []
    fpn_anchors_fid = np.zeros(0).astype(int)
    fpn_anchors = np.zeros([0, 4])
    fpn_labels = np.zeros(0)
    fpn_inds_inside = []
    for feat_id in range(len(feat_strides)):
        # len(scales.shape) == 1 just for backward compatibility, will remove in the future
        if len(scales.shape) == 1:
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios, scales=scales)
        else:
            assert len(scales.shape) == len(ratios.shape) == 2
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios[feat_id], scales=scales[feat_id])
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shapes[feat_id][-2:]

        # 1. generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, feat_width) * feat_strides[feat_id]
        shift_y = np.arange(0, feat_height) * feat_strides[feat_id]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors
        K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # for sigmoid classifier, ignore the 'background' class
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        fpn_anchors_fid = np.hstack((fpn_anchors_fid, len(inds_inside)))
        fpn_anchors = np.vstack((fpn_anchors, anchors))
        fpn_labels = np.hstack((fpn_labels, labels))
        fpn_inds_inside.append(inds_inside)
        fpn_args.append([feat_height, feat_width, A, total_anchors])

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(fpn_anchors.astype(np.float32, copy=False), gt_boxes.astype(np.float32, copy=False))
        # overlaps = iou(fpn_anchors.astype(np.float), gt_boxes.astype(np.float)[:, :4])
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(fpn_anchors)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        # fg label: for each gt, anchor with highest overlap
        fpn_labels[gt_argmax_overlaps] = 1
        # fg label: above threshold IoU
        fpn_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        fpn_labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(fpn_labels >= 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        fpn_labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else cfg.TRAIN.RPN_BATCH_SIZE - np.sum(fpn_labels >= 1)
    bg_inds = np.where(fpn_labels == 0)[0]
    fpn_anchors_fid = np.hstack((0, fpn_anchors_fid.cumsum()))

    if balance_scale_bg:
        num_bg_scale = num_bg / len(feat_strides)
        for feat_id in range(0, len(feat_strides)):
            bg_ind_scale = bg_inds[(bg_inds >= fpn_anchors_fid[feat_id]) & (bg_inds < fpn_anchors_fid[feat_id+1])]
            if len(bg_ind_scale) > num_bg_scale:
                disable_inds = npr.choice(bg_ind_scale, size=(len(bg_ind_scale) - num_bg_scale), replace=False)
                fpn_labels[disable_inds] = -1
    else:
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            if DEBUG:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            fpn_labels[disable_inds] = -1

    fpn_bbox_targets = np.zeros((len(fpn_anchors), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        fpn_bbox_targets[fpn_labels >= 1, :] = bbox_transform(fpn_anchors[fpn_labels >= 1, :], gt_boxes[argmax_overlaps[fpn_labels >= 1], :4])
        # fpn_bbox_targets[:] = bbox_transform(fpn_anchors, gt_boxes[argmax_overlaps, :4])
    # fpn_bbox_targets = (fpn_bbox_targets - np.array(cfg.TRAIN.BBOX_MEANS)) / np.array(cfg.TRAIN.BBOX_STDS)
    fpn_bbox_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)

    fpn_bbox_weights[fpn_labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    for feat_id in range(0, len(feat_strides)):
        feat_height, feat_width, A, total_anchors = fpn_args[feat_id]
        # map up to original set of anchors
        labels = _unmap(fpn_labels[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=-1)
        bbox_targets = _unmap(fpn_bbox_targets[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        bbox_weights = _unmap(fpn_bbox_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)

        labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, A * feat_height * feat_width))

        bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
        bbox_targets = bbox_targets.reshape((1, A * 4, -1))
        bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
        bbox_weights = bbox_weights.reshape((1, A * 4, -1))

        label_list.append(labels)
        bbox_target_list.append(bbox_targets)
        bbox_weight_list.append(bbox_weights)

    label = {
        'label': np.concatenate(label_list, axis=1),
        'bbox_target': np.concatenate(bbox_target_list, axis=2),
        'bbox_weight': np.concatenate(bbox_weight_list, axis=2)
    }

    return label
