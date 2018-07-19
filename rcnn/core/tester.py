from __future__ import print_function

import os
import time
import cv2
import random
import mxnet as mx
import numpy as np

from six.moves import cPickle, range
from .module import MutableModule
from .helper import get_scale_factor
from ..config import config
from ..config import config as cfg
from ..io import image
from ..processing.bbox_transform import nonlinear_pred, clip_boxes
from ..processing.nms import py_nms_wrapper

import matplotlib.pyplot as plt
from keypoint_utils import decode_keypoint, get_skeletons, get_keypoint_wrt_box

bbox_pred = nonlinear_pred


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))


def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    boxes = output['rpn_rois_output'].asnumpy()[:, :, 1:]
    scores = output['rpn_scores_output'].asnumpy()[:, :, 0:1]

    bs = boxes.shape[0]

    box_score = np.concatenate([boxes, scores], axis=2).tolist()
    for img_id in range(len(box_score)):
        for box_id in range(len(box_score[img_id])):
            box_score[img_id][box_id] = tuple(box_score[img_id][box_id])
    for img_id in range(len(box_score)):
        box_score[img_id] = list(set(box_score[img_id]))
        box_score[img_id] = np.array(box_score[img_id])
    scores = [single_img_box_score[:, 4] for single_img_box_score in box_score]
    boxes = [single_img_box_score[:, :4] for single_img_box_score in box_score]

    for i in range(len(scores)):
        order = np.argsort(-scores[i])
        boxes[i] = boxes[i][order]
        scores[i] = scores[i][order]

    for i in range(len(scores)):
        boxes[i] = boxes[i][:config.TRAIN.PROPOSAL_POST_NMS_TOP_N]
        scores[i] = scores[i][:config.TRAIN.PROPOSAL_POST_NMS_TOP_N][:, None]

    for i in range(len(boxes)):
        num_proposals = (boxes[i].shape)[0]
        if num_proposals < config.TRAIN.PROPOSAL_POST_NMS_TOP_N:
            pad_num = config.TRAIN.PROPOSAL_POST_NMS_TOP_N - num_proposals
            for idx in range(pad_num):
                rand_idx = np.random.randint(0, num_proposals)
                boxes[i] =  np.row_stack((boxes[i], boxes[i][rand_idx]))
                scores[i] =  np.row_stack((scores[i], scores[i][rand_idx]))

    boxes = np.array(boxes)
    scores = np.array(scores)

    # transform to original scale
    boxes = boxes / scale[:, None, None]

    return scores, boxes, data_dict


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[:, 2]
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        t2 = time.time() - t
        t = time.time()

        for ii in range(scores.shape[0] - data_batch.pad):
            # assemble proposals
            dets = np.concatenate((boxes[ii], scores[ii]), axis=-1)
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4] > thresh)
            dets = dets[keep]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)

            i += 1
        t3 = time.time() - t
        t = time.time()
        if i % 100 == 0:
            print('generating %d/%d' % (i + 1, imdb.num_images) + ' proposal %d' % (dets.shape[0]) +
                ' data %.4fs net %.4fs post %.4fs' % (t1, t2, t3))

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    return imdb_boxes, original_boxes


def im_detect_mask(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)
    data_dict = dict(zip(data_names, data_batch.data))
    # ['mask_roi_score', 'mask_prob_output', 'mask_roi_pred_boxes']
    if config.TEST.HAS_RPN:
        pred_boxes = output['mask_roi_pred_boxes'].asnumpy()
        scores = output['mask_roi_score'].asnumpy()
        mask_outputs = output['mask_prob_output'].asnumpy()
        mask_outputs = mask_outputs.reshape((data_batch.data[0].shape[0], -1) + mask_outputs.shape[1:])
    else:
        raise NotImplementedError
    # we used scaled image & roi to train, so it is necessary to transform them back
    if isinstance(scale, float) or isinstance(scale, int):
        pred_boxes = pred_boxes / scale
    elif isinstance(scale, np.ndarray):
        pred_boxes = pred_boxes / scale[:, None, None]

    return scores, pred_boxes, data_dict, mask_outputs


def im_detect_keypoint(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)
    # ['kp_prob_reshape_output', 'kp_roi_pred_boxes', 'kp_roi_score']
    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        pred_boxes = output['kp_roi_pred_boxes'].asnumpy()
        scores = output['kp_roi_score'].asnumpy()
        if config.KEYPOINT.USE_HEATMAP:
            kp_outputs = output['kp_prob_output'].asnumpy()
            kp_outputs = kp_outputs.reshape((data_batch.data[0].shape[0], -1) + kp_outputs.shape[1:])
        else:
            kp_outputs = output['kp_prob_reshape_output'].asnumpy()
    else:
        raise NotImplementedError
    # we used scaled image & roi to train, so it is necessary to transform them back
    if isinstance(scale, float) or isinstance(scale, int):
        pred_boxes = pred_boxes / scale
    elif isinstance(scale, np.ndarray):
        pred_boxes = pred_boxes / scale[:, None, None]

    return scores, pred_boxes, data_dict, kp_outputs


def pred_eval_mask(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    num_images, num_classes = imdb.num_images, imdb.num_classes

    results_list = []
    all_boxes = [[None for _ in range(num_images)] for _ in range(num_classes)]  # (#cls, #img)
    all_masks = [[None for _ in range(num_images)] for _ in range(num_classes)]  # (#cls, #img)

    img_ind = 0
    t = time.time()
    SCORE_THRESH = 0.05
    for im_infos, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = im_infos[:, 2]
        scores, boxes, data_dict, mask_outputs = im_detect_mask(predictor, data_batch, data_names, scales)

        t2 = time.time() - t
        t = time.time()

        for i in range(data_batch.data[0].shape[0] - data_batch.pad):
            score, box, mask_output, im_info = scores[i], boxes[i], mask_outputs[i], im_infos[i:i + 1]
            roi_rec = roidb[img_ind]
            label = np.argmax(score, axis=1)
            label = label[:, np.newaxis]


            for cls_ind in range(num_classes):
                cls_boxes = box[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_masks = mask_output[:, cls_ind, :, :]
                cls_scores = score[:, cls_ind, np.newaxis]
                keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
                cls_masks = cls_masks[keep, :, :]
                dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = nms(dets)
                all_boxes[cls_ind][img_ind] = dets[keep, :]
                all_masks[cls_ind][img_ind] = cls_masks[keep, :]

            """for j in range(1, num_classes):
                inds = np.where(score[:, j] > SCORE_THRESH)[0]
                cls_masks = mask_output[:, j, :, :]
                score_j = score[inds, j]
                box_j = box[inds, j * 4:(j + 1) * 4]
                dets_j = np.hstack((box_j, score_j[:, np.newaxis])).astype(np.float32, copy=False)
                cls_masks = cls_masks[inds, :, :]
                #nms
                keep = nms(dets_j)
                nms_dets = dets_j[keep, :]
                all_boxes[j][img_ind] = nms_dets
                all_masks[j][img_ind] = cls_masks[keep, :]"""

            # the first class is empty because it is background
            boxes_this_image = [[]] + [all_boxes[cls_ind][img_ind] for cls_ind in range(1, num_classes)]
            masks_this_image = [[]] + [all_masks[cls_ind][img_ind] for cls_ind in range(1, num_classes)]

            results_list.append({'image': roi_rec['image'],
                                 'im_info': im_info,
                                 'boxes': boxes_this_image,
                                 'masks': masks_this_image})
            t3 = time.time() - t
            t = time.time()
            if img_ind % 100 == 0:
                print('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(img_ind + 1, num_images, t1, t2, t3))
            img_ind += 1

    results_pack = {'all_boxes': all_boxes,
                    'all_masks': all_masks,
                    'results_list': results_list}

    imdb.evaluate_mask(results_pack)
    imdb.evaluate_detections(results_pack)


def pred_eval_keypoint(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    num_images, num_classes = imdb.num_images, imdb.num_classes

    results_list = []
    all_boxes = [[None for _ in range(num_images)] for _ in range(num_classes)]  # (#cls, #img)
    all_masks = [[None for _ in range(num_images)] for _ in range(num_classes)]  # (#cls, #img)
    all_kps = [[None for _ in range(num_images)] for _ in range(num_classes)]  # (#cls, #img)

    img_ind = 0
    t = time.time()
    SCORE_THRESH = 0.05
    for im_infos, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = im_infos[:, 2]
        scores, boxes, data_dict, kp_output = im_detect_keypoint(predictor, data_batch, data_names, scales)

        t2 = time.time() - t
        t = time.time()

        for i in range(data_batch.data[0].shape[0] - data_batch.pad):
            score, box, kp_output, im_info = scores[i], boxes[i], kp_output[i], im_infos[i:i + 1]
            roi_rec = roidb[img_ind]
            label = np.argmax(score, axis=1)
            label = label[:, np.newaxis]
            kp_pred = np.argmax(kp_output, axis=2)


            for cls_ind in range(num_classes):
                cls_boxes = box[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = score[:, cls_ind, np.newaxis]
                keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
                dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = nms(dets)
                all_boxes[cls_ind][img_ind] = dets[keep, :]
                all_kps[cls_ind][img_ind] = kp_pred[keep,:]


            # the first class is empty because it is background
            boxes_this_image = [[]] + [all_boxes[cls_ind][img_ind] for cls_ind in range(1, num_classes)]
            kps_this_image = [[]] + [all_kps[cls_ind][img_ind] for cls_ind in range(1, num_classes)]

            results_list.append({'image': roi_rec['image'],
                                 'im_info': im_info,
                                 'boxes': boxes_this_image,
                                 'keypoints': kps_this_image})
            t3 = time.time() - t
            t = time.time()
            if img_ind % 1 == 0:
                print('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(img_ind + 1, num_images, t1, t2, t3))
            img_ind += 1

    results_pack = {'all_boxes': all_boxes,
                    'all_kps': all_kps,
                    'results_list': results_list}

    imdb.evaluate_keypoints(results_pack)
    imdb.evaluate_detections(results_pack)


def pred_demo_mask(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = -1

    num_images = imdb.num_images
    num_classes = imdb.num_classes

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    img_ind = 0
    for im_info, data_batch in test_data:
        roi_rec = roidb[img_ind]
        scale = im_info[0, 2]
        scores, boxes, data_dict, mask_output = im_detect_mask(predictor, data_batch, data_names, scale)
        scores, boxes, mask_output = scores[0], boxes[0], mask_output[0]

        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        all_masks = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]

        for cls_ind in range(num_classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind] = dets[keep, :]
            all_masks[cls_ind] = cls_masks[keep, :, :]

        # the first class is empty because it is background
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, num_classes)]
        masks_this_image = [[]] + [all_masks[j] for j in range(1, num_classes)]
        filename = roi_rec['image'].split("/")[-1]
        filename = result_path + '/' + filename.replace('.png', '.jpg')
        data_dict = dict(zip(data_names, data_batch.data))
        draw_detection_mask(data_dict['data'], boxes_this_image, masks_this_image, 1.0, filename, imdb.classes)
        # draw_detection(data_dict['data'], boxes_this_image, 1.0, filename)
        img_ind += 1

def pred_demo_keypoint(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = -1

    num_images = imdb.num_images
    num_classes = imdb.num_classes

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    img_ind = 0
    for im_info, data_batch in test_data:
        roi_rec = roidb[img_ind]
        scale = im_info[0, 2]
        scores, boxes, data_dict, kp_output = im_detect_keypoint(predictor, data_batch, data_names, scale)
        scores, boxes, kp_output = scores[0], boxes[0], kp_output[0]

        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        # all_masks = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        all_kps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]
        kp_pred = np.argmax(kp_output, axis=2)

        for cls_ind in range(num_classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            # cls_masks = mask_output[:, cls_ind, :, :]
            # cls_kps = kp_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            # cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind] = dets[keep, :]
            # all_masks[cls_ind] = cls_masks[keep, :, :]

            # keypoints for person only
            # if cls_ind == 1:
            all_kps[cls_ind] = kp_pred[keep,:]

        # the first class is empty because it is background
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, num_classes)]
        # masks_this_image = [[]] + [all_masks[j] for j in range(1, num_classes)]
        kps_this_image = [[]] + [all_kps[j] for j in range(1, num_classes)]
        filename = roi_rec['image'].split("/")[-1]
        filename = result_path + '/' + filename.replace('.png', '.jpg')
        data_dict = dict(zip(data_names, data_batch.data))
        draw_detection_keypoint(data_dict['data'], boxes_this_image, kps_this_image, 1.0, filename, imdb.classes)
        # draw_detection(data_dict['data'], boxes_this_image, 1.0, filename)
        img_ind += 1

def pred_demo_keypoint_heatmap(predictor, test_data, imdb, roidb, result_path, vis=False, thresh=1e-1):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = -1

    num_images = imdb.num_images
    num_classes = imdb.num_classes

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    img_ind = 0
    for im_info, data_batch in test_data:
        roi_rec = roidb[img_ind]
        scale = im_info[0, 2]
        scores, boxes, data_dict, kp_output = im_detect_keypoint(predictor, data_batch, data_names, scale)
        scores, boxes, kp_output = scores[0], boxes[0], kp_output[0]
        # kp_output(1, 1000, 17, 28, 28)

        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        all_kps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]
        # kp_pred = np.argmax(kp_output, axis=2)

        for cls_ind in range(num_classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_kps = kp_output[keep, :, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind] = dets[keep, :]
            all_kps[cls_ind] = cls_kps[keep, :, :, :]

            # keypoints for person only
            # if cls_ind == 1:
            # all_kps[cls_ind] = kp_pred[keep,:]

        # the first class is empty because it is background
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, num_classes)]
        kps_this_image = [[]] + [all_kps[j] for j in range(1, num_classes)]
        filename = roi_rec['image'].split("/")[-1]
        filename = result_path + '/' + filename.replace('.png', '.jpg')
        data_dict = dict(zip(data_names, data_batch.data))
        draw_detection_keypoint_heatmap_binary(data_dict['data'], boxes_this_image, kps_this_image, 1.0, filename, imdb.classes)
        # draw_detection(data_dict['data'], boxes_this_image, 1.0, filename)
        img_ind += 1

def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=8, color='white')
    # plt.show()
    save_name = 'output/rpn/{}.jpg'.format(random.randint(0, 10000))
    print(save_name)
    plt.axis('off')
    plt.savefig(save_name)
    plt.close('all')


def draw_detection_mask(im_array, boxes_this_image, masks_this_image, scale, filename, class_names):
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = boxes_this_image[j]
        masks = masks_this_image[j]
        print('len(dets):', len(dets))
        for i in range(len(dets)):
            bbox = dets[i, :4] * scale
            score = dets[i, -1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            mask = masks[i, :, :]
            mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask_color = random.randint(80, 100)
            c = random.randint(0, 2)
            target = im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] + mask_color * mask
            target[target >= 255] = 255
            im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] = target
    print(filename)
    cv2.imwrite(filename, im)

def draw_detection_keypoint_heatmap_binary(im_array, boxes_this_image, kps_this_image, scale, filename, class_names):
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    plt.imshow(im)

    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        dets = boxes_this_image[j]
        kps_heatmap = kps_this_image[j]

        print('len(dets):', len(dets))
        for i in range(len(dets)):
            color = np.random.random(size=(3,))
            bbox = dets[i, :4] * scale
            score = dets[i, -1]
            bbox = map(int, bbox)
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1]+10, '%s %.3f' % (class_names[j], score), bbox={'facecolor':'white', 'alpha':0.5},
                     size='x-small')

            # kps_heatmap[kps_heatmap[:,:,:] > 0.5] = 1
            # kps_heatmap[kps_heatmap[:,:,:] <= 0.5] = 0

            kps_heatmap_single = kps_heatmap[i]
            xs, ys = np.zeros(17, dtype=np.int32), np.zeros(17, dtype=np.int32)
            for k in range(5,6):
                kps_heatmap_single_resize = cv2.resize(kps_heatmap_single[k], (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                kps_heatmap_single_resize[kps_heatmap_single_resize>0.5] = 1
                kps_heatmap_single_resize[kps_heatmap_single_resize<=0.5] = 0

                mask_color = random.randint(80, 100) / 255.0
                c = random.randint(0, 2)
                target = im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] + mask_color * kps_heatmap_single_resize
                target[target >= 1] = 1
                im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] = target


                xtemp, ytemp = np.unravel_index(kps_heatmap_single[k].argmax(), kps_heatmap_single[k].shape)
                xs[k], ys[k] = get_keypoint_wrt_box(xtemp, ytemp, bbox)
                plt.plot(xs[k], ys[k], 'o', color=color)
            # skeletons = get_skeletons()
            # for sk in skeletons:
            #     plt.plot(xs[sk], ys[sk], linewidth=2, color=color, alpha=0.5)

    print(filename)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=199)
    plt.close('all')


def draw_detection_keypoint(im_array, boxes_this_image, kps_this_image, scale, filename, class_names):
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    plt.imshow(im)

    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        dets = boxes_this_image[j]
        kps = kps_this_image[j]
        print('len(dets):', len(dets))
        for i in range(len(dets)):
            color = np.random.random(size=(3,))
            bbox = dets[i, :4] * scale
            score = dets[i, -1]
            bbox = map(int, bbox)
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1]+10, '%s %.3f' % (class_names[j], score), bbox={'facecolor':'white', 'alpha':0.5},
                     size='x-small')
            keypoint = kps[i]
            xs, ys = np.zeros(17, dtype=np.int32), np.zeros(17, dtype=np.int32)
            for k, kp in enumerate(keypoint):
                xs[k], ys[k] = decode_keypoint(kp, bbox)
                plt.plot(xs[k], ys[k], 'o', color=color)
            skeletons = get_skeletons()
            for sk in skeletons:
                plt.plot(xs[sk], ys[sk], linewidth=2, color=color, alpha=0.5)

    print(filename)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=199)
    plt.close('all')

            # bbox = dets[i, :4] * scale
            # score = dets[i, -1]
            # bbox = map(int, bbox)
            # cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            # cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
            #             color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            # mask = masks[i, :, :]
            # mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
            # mask[mask > 0.5] = 1
            # mask[mask <= 0.5] = 0
            # mask_color = random.randint(80, 100)
            # c = random.randint(0, 2)
            # target = im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] + mask_color * mask
            # target[target >= 255] = 255
            # im[bbox[1]: bbox[3], bbox[0]: bbox[2], c] = target
    # cv2.imwrite(filename, im)

def draw_detection_keypoint_heatmap(im_array, boxes_this_image, kps_this_image, scale, filename, class_names):
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    plt.imshow(im)

    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        dets = boxes_this_image[j]
        kps_heatmap = kps_this_image[j]

        print('len(dets):', len(dets))
        for i in range(len(dets)):
            color = np.random.random(size=(3,))
            bbox = dets[i, :4] * scale
            score = dets[i, -1]
            bbox = map(int, bbox)
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1]+10, '%s %.3f' % (class_names[j], score), bbox={'facecolor':'white', 'alpha':0.5},
                     size='x-small')
            kps_heatmap_single = kps_heatmap[i]
            xs, ys = np.zeros(17, dtype=np.int32), np.zeros(17, dtype=np.int32)
            for k in range(kps_heatmap_single.shape[0]):
                xtemp, ytemp = np.unravel_index(kps_heatmap_single[k].argmax(), kps_heatmap_single[k].shape)
                xs[k], ys[k] = get_keypoint_wrt_box(xtemp, ytemp, bbox)
                plt.plot(xs[k], ys[k], 'o', color=color)
            skeletons = get_skeletons()
            for sk in skeletons:
                plt.plot(xs[sk], ys[sk], linewidth=2, color=color, alpha=0.5)

    print(filename)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=199)
    plt.close('all')

def draw_detection(im_array, boxes_this_image, scale, filename):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    class_names = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = boxes_this_image[j]
        print ('num of detected boxes:', len(dets))
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    print(filename)
    cv2.imwrite(filename, im)
