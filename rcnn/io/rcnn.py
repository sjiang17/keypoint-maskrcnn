"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""
from __future__ import print_function

import cv2
import numpy as np
import numpy.random as npr

from ..config import config
from ..config import config as cfg
# from ..core.segms import polys_to_mask_wrt_box
from ..core.keypoint_utils import keypoints_to_vec_wrt_box
from ..core.helper import get_scale_factor
from ..io.image import get_image, get_image_and_mask, tensor_vstack
from ..processing.bbox_transform import bbox_overlaps, nonlinear_transform
from ..processing.bbox_regression import expand_bbox_regression_targets

bbox_transform = nonlinear_transform

def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys


def get_fpn_maskrcnn_batch(roidb, max_shape):
    """
    return a dictionary that contains raw data.
    """
    assert config.TRAIN.BATCH_ROIS % config.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_ROIS)

    # (optionally) resize image, bbox, semantic segmentation and instance segmentation all together
    imgs, roidb = \
        get_image(roidb,
                  use_random_scale=config.TRAIN.SCALE,
                  random_scale_range=config.TRAIN.SCALE_RANGE,
                  fixed_multi_scales=config.SCALES,
                  pixel_mean=config.PIXEL_MEANS)
    # print('max_shape:', max_shape)
    # print('imgs:', len(imgs))
    im_array = tensor_vstack(imgs, shape=max_shape)

    # sample rois
    rois_per_image = config.TRAIN.BATCH_ROIS / config.TRAIN.BATCH_IMAGES
    fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    rois_list = {"stride%s" % s: list() for s in config.RCNN_FEAT_STRIDE}
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    # mask_target_list = list()
    consist_target_list = list()
    keypoint_target_list = list()

    batch_size = len(roidb)  # batch_size = 1 for 1 gpu
    num_classes = config.NUM_CLASSES
    for i in range(batch_size):
        roi_rec = roidb[i]
        # label = class RoI has max overlap with
        rois = roi_rec['boxes'] # (2001,4)
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']
        im_info = roi_rec['im_info'] # [height, width, scale] [800, 1199, 1.87]
        kp_ids = roi_rec['kp_id']  # shape = (2001,)
        # print(kp_ids)
        keypoints = roi_rec['keypoints']
        # print(keypoints)

        # if 'ins_poly' in roi_rec:
        #     # coco style
        #     ins_ids = roi_rec['ins_id']
        #     ins_polys = roi_rec['ins_poly']
        #     mask_targets = None
        #     mask_labels = None
        #     mask_inds = None
        # else:
        #     # cityscape style
        #     ins_ids = None
        #     ins_polys = None
        #     mask_targets = roi_rec['mask_targets']
        #     mask_labels = roi_rec['mask_labels']
        #     mask_inds = roi_rec['mask_inds']

        rois_on_lvls, labels, bbox_targets, bbox_weights, keypoint_targets = \
            sample_rois_fpn(
                            roidb[i],
                            rois,
                            fg_rois_per_image,
                            rois_per_image,
                            num_classes,
                            labels,
                            overlaps,
                            bbox_targets,
                            # mask_targets=mask_targets,
                            # mask_labels=mask_labels,
                            # mask_inds=mask_inds,
                            # ins_ids=ins_ids,
                            # ins_polys=ins_polys,
                            im_info=im_info,
                            kp_ids = kp_ids,
                            keypoints = keypoints
                            )
        #print (mask_targets.shape)
        #np.save("mask_targets", mask_targets)

        # transform sampled data
        for k in rois_list:
            # TODO: only support 1 image per GPU now, batch_idx MUST be per-card.
            rois_with_batch_idx = np.concatenate([np.full((rois_per_image, 1), 0), rois_on_lvls[k]], axis=1)
            rois_with_batch_idx = np.expand_dims(rois_with_batch_idx, axis=0)
            rois_list[k].append(rois_with_batch_idx)

        label_list.append(np.expand_dims(labels, axis=0))
        bbox_target_list.append(np.expand_dims(bbox_targets, axis=0))
        bbox_weight_list.append(np.expand_dims(bbox_weights, axis=0))
        # mask_target_list.append(np.expand_dims(mask_targets, axis=0))
        keypoint_target_list.append(np.expand_dims(keypoint_targets, axis=0))



    data_dict = {"data": im_array}
    data_dict.update(
        {"rois_stride%s" % s: np.concatenate(rois_list["stride%s" % s]) for s in config.RCNN_FEAT_STRIDE})

    label_dict = {"label": np.concatenate(label_list),
                  "bbox_target": np.concatenate(bbox_target_list),
                  "bbox_weight": np.concatenate(bbox_weight_list),
                  # "mask_target": np.concatenate(mask_target_list),
                  "keypoint_target": np.concatenate(keypoint_target_list)
                  }

    return data_dict, label_dict


def sample_rois_fpn(roidb, rois, fg_rois_per_image, rois_per_image, num_classes, labels=None, overlaps=None,
                    bbox_targets=None, im_info=None, consist_targets=None, consist_labels=None, consist_inds=None,
                    kp_ids = None, keypoints = None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param roidb: roidb extended format [image_index]
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param assign_levels: [n]
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :return: (rois_on_levels, labels, bbox_targets, bbox_weights)
    """
    DEBUG = False

    if DEBUG:
        print("rois.shape: {}".format(rois.shape))
        print("labels.shape: {}".format(labels.shape))

    num_rois = rois.shape[0]
    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    # minimum number between 128 and fg
    fg_rois_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)

    if DEBUG:
        print('fg total num: {}'.format(len(fg_indexes)))

    # Sample foreground regions without replacement
    # if fg number > 128, then random choose 128 fg from all fg.
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    if DEBUG:
        print('bg total num: {}'.format(len(bg_indexes)))
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    # bg num = 512 - fg num
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    # minimum number between calculated bg number and actual bg number
    bg_rois_this_image = np.minimum(bg_rois_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    # if actuall bg number > calculated bg number, randomly choose bg
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_this_image, replace=False)
    if DEBUG:
        print('fg num: {}'.format(len(fg_indexes)))
        print('bg num: {}'.format(len(bg_indexes)))

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]

    # pad more to ensure a fixed minibatch size
    # if fg + bg is less than 512, means not enough training example, need to pad more
    # pad negative rois, which are randomly chosen from rois whose overlaps less than FG Threshold.
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select fg labels
    labels = labels[keep_indexes]
    labels[fg_rois_this_image:] = 0  # set labels of bg_rois to be 0

    # select fg rois
    rois = rois[keep_indexes]

    # config.RCNN_FEAT_STRIDE = [32, 16, 8, 4]
    # assign rois to different strides (level of feature maps)
    thresholds = [[np.inf, 448], [448, 224], [224, 112], [112, 0]]
    rois_area = np.sqrt((rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
    assign_levels = np.zeros(rois_per_image, dtype=np.uint8)
    for thresh, stride in zip(thresholds, config.RCNN_FEAT_STRIDE):
        inds = np.logical_and(thresh[1] <= rois_area, rois_area < thresh[0])
        assign_levels[inds] = stride

    # bg rois statistics
    if DEBUG:
        bg_assign = assign_levels[bg_indexes]
        bg_rois_on_levels = dict()
        for i, s in enumerate(config.RCNN_FEAT_STRIDE):
            bg_rois_on_levels.update({'stride%s' % s: len(np.where(bg_assign == s)[0])})
        print(bg_rois_on_levels)

    # assign rois to levels
    rois_on_levels = dict()
    mask_rois_on_levels = None
    for i, s in enumerate(config.RCNN_FEAT_STRIDE):
        # find the index of rois whose assigned level is s
        index = np.where(assign_levels == s)
        # construct a zero array, shape is 512 * 4
        _rois = np.zeros(shape=(rois_per_image, 4), dtype=np.float32)
        # only rois in this level s are assigned value
        _rois[index] = rois[index]
        # same for mask, only rois in this level are assigned value 1
        _rois_mask = np.zeros(shape=(rois_per_image,), dtype=np.float32)
        _rois_mask[index[0]] = 1
        rois_on_levels.update({"stride%s" % s: _rois})

    # load bbox_target
    bbox_target_data = bbox_targets[keep_indexes, :]
    bbox_targets, bbox_weights = expand_bbox_regression_targets(bbox_target_data, num_classes)

    # load keypoint target
    kp_ids_new = kp_ids[keep_indexes][:fg_rois_per_image]
    keypoint_targets = np.full(fill_value=-1, shape=(fg_rois_per_image, 17), dtype=np.int32)
    for i in range(len(kp_ids_new)):
        keypoint_targets[i] = keypoints_to_vec_wrt_box(keypoints[kp_ids_new[i]], rois[i] / im_info[2], config.KEYPOINT.MAPSIZE)

    # # load mask target
    # if mask_targets is not None and mask_labels is not None and mask_inds is not None:
    #     mask_targets = _mask_scatter(mask_targets, mask_labels, mask_inds, num_rois, num_classes)
    #     mask_targets = mask_targets[keep_indexes][:fg_rois_per_image]
    # elif ins_ids is not None and ins_polys is not None:
    #     # COCO data set goes here ...
    #
    #     # instance ids are those kept fg rois.
    #     ins_ids = ins_ids[keep_indexes][:fg_rois_per_image]
    #     # fill the mask target with -1. shape = (128 * num_class * 28 * 28)
    #     mask_targets = np.full(fill_value=-1, shape=(fg_rois_per_image, num_classes, 28, 28), dtype=np.float32)
    #
    #     # for each fg roi, (real fg rois, since it may be less than 128)
    #     for i in range(fg_rois_this_image):                                         # im_info[2] is image_scale
    #         mask_targets[i, labels[i]] = polys_to_mask_wrt_box(ins_polys[ins_ids[i]], rois[i] / im_info[2], 28)
    #
        # #polys_gt_inds = np.where((roidb['gt_classes'] > 0))[0]
        # #polys_gt = [roidb['ins_poly'][i] for i in polys_gt_inds]
        # #boxes_from_polys = polys_to_boxes(polys_gt)
        # #mask_targets = np.full(fill_value=-1, shape=(fg_rois_per_image, num_classes, 28, 28), dtype=np.float32)
        # ## Find overlap between all foreground rois and the bounding boxes
        # ## enclosing each segmentation
        # #rois_fg = rois[:fg_rois_this_image]
        # #overlaps_bbfg_bbpolys = bbox_overlaps(
        # #    rois_fg.astype(np.float32, copy=False),
        # #    boxes_from_polys.astype(np.float32, copy=False)
        # #)
        # ## Map from each fg rois to the index of the mask with highest overlap
        # ## (measured by bbox overlap)
        # #fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)
        # ##print (fg_polys_inds)
        # ## add fg targets
        # #im_name = roidb["image"]
        # #for i in range(rois_fg.shape[0]):
        # #    fg_polys_ind = fg_polys_inds[i]
        # #    poly_gt = polys_gt[fg_polys_ind]
        # #    roi_fg = rois_fg[i]
        # #    im_ori = cv2.imread(im_name)
        # #    im_name_cur = "./mask_ins/" + (im_name.split("/"))[-1][:-4] + "_" + str(i) + ".jpg"
        # #    roi_cur = roi_fg / im_info[2]
        # #    cv2.rectangle(im_ori, (int(roi_cur[0]),int(roi_cur[1])), (int(roi_cur[2]),int(roi_cur[3])), (0,255,0), 1)
        # #    #pts1 = np.array([[roi_cur[0],roi_cur[1]],[roi_cur[2],roi_cur[3]]], np.int32)
        # #    #pts1 = pts1.reshape((-1,1,2))
        # #    for item_ins in poly_gt:
        # #        seq = []
        # #        for idx in range(len(item_ins)/2):
        # #            x = item_ins[idx * 2]
        # #            y = item_ins[idx * 2 + 1]
        # #            seq.append([x,y])
        # #        pts = np.array(seq, np.int32)
        # #        pts = pts.reshape((-1,1,2))
        # #        cv2.polylines(im_ori,[pts],True,(0,255,255))
        # #    cv2.imwrite(im_name_cur, im_ori)
        # #    # Rasterize the portion of the polygon mask within the given fg roi
        # #    # to an M x M binary image
        # #    mask = polys_to_mask_wrt_box(poly_gt, roi_fg, 28)
        # #    mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
        # #    #masks[i, :] = np.reshape(mask, M**2)
        # #    mask_targets[i, labels[i]] = np.reshape(mask, (28,28))

    # if mask_targets is not None:
    #     return rois_on_levels, labels, bbox_targets, bbox_weights, mask_targets, keypoint_targets
    # else:
    #     return rois_on_levels, labels, bbox_targets, bbox_weights

    return rois_on_levels, labels, bbox_targets, bbox_weights, keypoint_targets


def _mask_scatter(mask_targets, mask_labels, mask_inds, num_rois, num_classes):
    mr = mask_targets.shape[-1]  # mask_resolution
    _mask_targets = np.full(shape=(num_rois, num_classes, mr, mr), fill_value=-1, dtype=np.int8)
    _mask_targets[mask_inds, mask_labels] = mask_targets
    return _mask_targets  # [#rois, #classes, mr, mr]


def rescale_mask(masks, labels, rescales, im_info):
    ret_masks = np.full(masks.shape, fill_value=-1, dtype=np.int8)
    for i in range(len(masks)):
        cls = labels[i]
        mask = masks[i][cls]
        scale = rescales[i]
        x_inds = np.where(mask[0] == -1)[0]
        y_inds = np.where(mask[:, 0] == -1)[0]
        w = x_inds[0] if len(x_inds) > 0 else mask.shape[0]
        h = y_inds[0] if len(y_inds) > 0 else mask.shape[1]
        if w > 0 and h > 0:  # ignore oversized rois which have all -1 masks
            mask = mask[0:h, 0:w].astype(np.uint8, copy=False)  # opencv can not resize int8 image
            mask = np.array(cv2.resize(mask, dsize=None, fx=scale, fy=scale) > 0.5, dtype=np.int8)
            new_h, new_w = mask.shape
            # if new_h > 112 or new_w > 112:
            #    print mask.shape
            new_h, new_w = np.minimum(new_h, 112), np.minimum(new_w, 112)
            ret_masks[i][cls][0:new_h, 0:new_w] = mask[0:new_h, 0:new_w]
    return ret_masks
