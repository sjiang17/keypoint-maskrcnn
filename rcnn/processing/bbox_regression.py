"""
This file has functions about generating bounding box regression targets
"""
from __future__ import print_function

import numpy as np
import cv2
import threading
import multiprocessing as mp
from six.moves import queue
from bbox_transform import bbox_overlaps, nonlinear_transform
from rcnn.config import config


bbox_transform = nonlinear_transform


def compute_bbox_regression_targets(rois, overlaps, labels):
    """
    given rois, overlaps, gt labels, compute bounding box regression targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)
    # Sanity check
    assert len(rois) == len(overlaps), '#rois != #max_overlaps'

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    assert len(gt_inds) > 0, 'zero ground truth rois'
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= config.TRAIN.BBOX_REGRESSION_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :].astype(np.float32, copy=False),
                                   rois[gt_inds, :].astype(np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets


def add_bbox_regression_targets(roidb):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    print('add bounding box regression targets')
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)
    num_classes = config.NUM_CLASSES
    for im_i in range(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = compute_bbox_regression_targets(rois, max_overlaps, max_classes)

    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # use fixed / precomputed means and stds instead of empirical values
        means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (num_classes, 1))
        stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (num_classes, 1))
    else:
        # compute mean, std values
        class_counts = np.zeros((num_classes, 1)) + 1e-14
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in range(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in range(1, num_classes):
                cls_indexes = np.where(targets[:, 0] == cls)[0]
                if cls_indexes.size > 0:
                    class_counts[cls] += cls_indexes.size
                    sums[cls, :] += targets[cls_indexes, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_indexes, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # var(x) = E(x^2) - E(x)^2
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # normalized targets
    for im_i in range(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in range(1, num_classes):
            cls_indexes = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] /= stds[cls, :]

    return means.ravel(), stds.ravel()


def compute_mask_and_label(rois, instances, labels, ins_seg, flipped):
    if isinstance(ins_seg, str):
        ins_seg = cv2.imread(ins_seg, -1)
    if flipped:
        ins_seg = ins_seg[:, ::-1]
    n_rois = rois.shape[0]
    class_id = config.CLASS_ID
    mask_target = np.zeros((n_rois, 28, 28), dtype=np.int8)
    mask_label = np.zeros((n_rois, ), dtype=np.int8)

    # YuntaoChen: hack for double resize overflow
    rois[:, 0][rois[:, 0] < 0] = 0
    for n in range(n_rois):
        target = ins_seg[int(rois[n, 1]): int(rois[n, 3]), int(rois[n, 0]): int(rois[n, 2])]
        ins_id = config.SEG_CODE * class_id[labels[n]] + instances[n]
        # YuntaoChen: hack for roi less than 1px
        if 0 not in target.shape:
            mask = np.zeros(target.shape)
        else:
            mask = np.zeros((1, 1))
        idx = np.where(target == ins_id)
        mask[idx] = 1
        mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_NEAREST)

        mask_target[n] = mask
        mask_label[n] = labels[n]
    return mask_target, mask_label


def compute_mask_and_label_fcn(rois, instances, labels, ins_seg, flipped):
    ins_seg_lvl = []
    if isinstance(ins_seg, str):
        ins_seg = cv2.imread(ins_seg, -1)
    ins_seg_lvl.append(ins_seg)
    ins_seg_lvl.append(cv2.resize(ins_seg, dsize=None, fx=1.0/2, fy=1.0/2, interpolation=cv2.INTER_NEAREST))
    ins_seg_lvl.append(cv2.resize(ins_seg, dsize=None, fx=1.0/4, fy=1.0/4, interpolation=cv2.INTER_NEAREST))
    ins_seg_lvl.append(cv2.resize(ins_seg, dsize=None, fx=1.0/8, fy=1.0/8, interpolation=cv2.INTER_NEAREST))

    if flipped:
        for ins_seg in ins_seg_lvl:
            ins_seg[...] = ins_seg[:, ::-1]

    n_rois = rois.shape[0]
    class_id = config.CLASS_ID
    mask_target = np.zeros((n_rois, 112, 112), dtype=np.int8)
    mask_label = np.zeros((n_rois, ), dtype=np.int8)
    for n in range(n_rois):
        x1, y1, x2, y2 = rois[n]
        long_side = max(x2 - x1, y2 - y1)
        if long_side <= 112:
            ins_seg = ins_seg_lvl[0]
        elif long_side <= 224:
            ins_seg = ins_seg_lvl[1]
            x1, y1, x2, y2 = x1/2, y1/2, x2/2, y2/2
        elif long_side <= 448:
            ins_seg = ins_seg_lvl[2]
            x1, y1, x2, y2 = x1/4, y1/4, x2/4, y2/4
        elif long_side <= 896:
            ins_seg = ins_seg_lvl[3]
            x1, y1, x2, y2 = x1/8, y1/8, x2/8, y2/8
        else:
            # do not handle very large instance for now
            ins_seg = ins_seg_lvl[0]
            x1, y1, x2, y2 = 0, 0, 0, 0

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        target = ins_seg[y1:y2, x1:x2]
        new_ins_id = config.SEG_CODE * class_id[labels[n]] + instances[n]
        mask = np.full(fill_value=-1, shape=(112, 112), dtype=np.int8)
        mask[0:(y2-y1), 0:(x2-x1)] = 0
        idx = np.where(target == new_ins_id)
        mask[idx] = 1

        mask_target[n] = mask
        mask_label[n] = labels[n]
    return mask_target, mask_label


def compute_bbox_mask_targets_and_label(rois, instances, overlaps, labels, seg, flipped, for_maskfcn):
    """
    given rois, overlaps, gt labels, seg, compute bounding box mask targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float32, copy=False)

    # Sanity check
    assert len(rois) == len(overlaps), 'number of proposal ROIs and max overlap with gt bbox does not match'

    fg_indexes = np.where(overlaps >= config.TRAIN.BBOX_REGRESSION_THRESH)[0]
    fg_rois = rois[fg_indexes, :]

    if for_maskfcn:
        mask_targets, mask_label = \
            compute_mask_and_label_fcn(fg_rois, instances[fg_indexes], labels[fg_indexes], seg, flipped)
    else:
        mask_targets, mask_label = \
            compute_mask_and_label(fg_rois, instances[fg_indexes], labels[fg_indexes], seg, flipped)
    return mask_targets, mask_label, fg_indexes


def add_mask_targets(roidb, for_maskfcn=False):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    if for_maskfcn:
        print('add bounding box mask targets for maskfcn')
    else:
        print('add bounding box mask targets for maskrcnn')

    num_images = len(roidb)

    # Multi threads processing
    im_quene = queue.Queue(maxsize=0)
    for im_i in range(num_images):
        im_quene.put(im_i)

    def process():
        while not im_quene.empty():
            im_i = im_quene.get()
            if im_i > 0 and im_i % 500 == 0:
                print("-----process img {}".format(im_i))
            rois = roidb[im_i]['boxes']
            max_instances = roidb[im_i]['ins_id']
            max_overlaps = roidb[im_i]['max_overlaps']
            max_classes = roidb[im_i]['max_classes']
            ins_seg = roidb[im_i]['ins_seg']
            flipped = roidb[im_i]['flipped']

            # gather masks for fore ground rois only
            # masks are later reconstructed in sample_rois using
            # mask_targets + mask_labels + mask_inds
            roidb[im_i]['mask_targets'], roidb[im_i]['mask_labels'], roidb[im_i]['mask_inds'] = \
                compute_bbox_mask_targets_and_label(rois, max_instances, max_overlaps, max_classes, ins_seg, flipped,
                                                    for_maskfcn)

    threads = [threading.Thread(target=process, args=()) for _ in range(mp.cpu_count())]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def add_consistent_targets(roidb):
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)

    # Multi threads processing
    im_quene = queue.Queue(maxsize=0)
    for im_i in range(num_images):
        im_quene.put(im_i)

    def process():
        while not im_quene.empty():
            im_i = im_quene.get()
            if im_i > 0 and im_i % 500 == 0:
                print("-----process img {}".format(im_i))
            rois = roidb[im_i]['boxes']
            max_overlaps = roidb[im_i]['max_overlaps']
            max_classes = roidb[im_i]['max_classes']
            sem_seg = roidb[im_i]['sem_seg']
            flipped = roidb[im_i]['flipped']

            fg_inds = np.where(max_overlaps >= config.TRAIN.BBOX_REGRESSION_THRESH)[0]
            roidb[im_i]['consist_inds'] = fg_inds

            # gather masks for fore ground rois only
            # masks are later reconstructed in sample_rois using
            # mask_targets + mask_labels + mask_inds
            roidb[im_i]['consist_targets'], roidb[im_i]['consist_labels'],  = \
                compute_consist_mask_and_label_fcn(rois[fg_inds], max_classes[fg_inds], sem_seg, flipped)

    threads = [threading.Thread(target=process, args=()) for _ in range(mp.cpu_count())]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def compute_consist_mask_and_label_fcn(rois, labels, sem_seg, flipped):
    sem_seg_lvl = []
    if isinstance(sem_seg, str):
        sem_seg = cv2.imread(sem_seg, -1)
    sem_seg_lvl.append(sem_seg)
    sem_seg_lvl.append(cv2.resize(sem_seg, dsize=None, fx=1.0 / 2, fy=1.0 / 2, interpolation=cv2.INTER_NEAREST))
    sem_seg_lvl.append(cv2.resize(sem_seg, dsize=None, fx=1.0 / 4, fy=1.0 / 4, interpolation=cv2.INTER_NEAREST))
    sem_seg_lvl.append(cv2.resize(sem_seg, dsize=None, fx=1.0 / 8, fy=1.0 / 8, interpolation=cv2.INTER_NEAREST))

    if flipped:
        for sem_seg in sem_seg_lvl:
            sem_seg[...] = sem_seg[:, ::-1]

    n_rois = rois.shape[0]
    class_id = config.CLASS_ID
    mask_target = np.zeros((n_rois, 112, 112), dtype=np.int8)
    mask_label = np.zeros((n_rois,), dtype=np.int8)
    for n in range(n_rois):
        x1, y1, x2, y2 = rois[n]
        long_side = max(x2 - x1, y2 - y1)
        if long_side <= 112:
            sem_seg = sem_seg_lvl[0]
        elif long_side <= 224:
            sem_seg = sem_seg_lvl[1]
            x1, y1, x2, y2 = x1 / 2, y1 / 2, x2 / 2, y2 / 2
        elif long_side <= 448:
            sem_seg = sem_seg_lvl[2]
            x1, y1, x2, y2 = x1 / 4, y1 / 4, x2 / 4, y2 / 4
        elif long_side <= 896:
            sem_seg = sem_seg_lvl[3]
            x1, y1, x2, y2 = x1 / 8, y1 / 8, x2 / 8, y2 / 8
        else:
            # do not handle very large instance for now
            sem_seg = sem_seg_lvl[0]
            x1, y1, x2, y2 = 0, 0, 0, 0

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        target = sem_seg[y1:y2, x1:x2]
        cls_id = class_id[labels[n]]
        mask = np.full(fill_value=-1, shape=(112, 112), dtype=np.int8)
        mask[0:(y2 - y1), 0:(x2 - x1)] = 0
        idx = np.where(target == cls_id)
        mask[idx] = -1

        mask_target[n] = mask
        mask_label[n] = labels[n]
    return mask_target, mask_label


def expand_bbox_regression_targets(bbox_targets_data, num_classes):
    """
    expand from 5 to 4 * num_classes; only the right class has non-zero bbox regression targets
    :param bbox_targets_data: [k * 5]
    :param num_classes: number of classes
    :return: bbox target processed [k * 4 num_classes]
    bbox_weights ! only foreground boxes have bbox regression computation!
    """
    classes = bbox_targets_data[:, 0]
    bbox_targets = np.zeros((classes.size, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    indexes = np.where(classes > 0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[index, start:end] = bbox_targets_data[index, 1:]
        bbox_weights[index, start:end] = config.TRAIN.BBOX_WEIGHTS
    return bbox_targets, bbox_weights
