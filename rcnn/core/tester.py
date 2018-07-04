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
        #print ("boxes[%s] shape:%s"%(i, str(boxes[i].shape)))
        #print ("scores[%s] shape:%s"%(i, str(scores[i].shape)))
        num_proposals = (boxes[i].shape)[0]
        #np.save("boxes", boxes[i])
        #np.save("scores", scores[i])
        if num_proposals < config.TRAIN.PROPOSAL_POST_NMS_TOP_N:
            pad_num = config.TRAIN.PROPOSAL_POST_NMS_TOP_N - num_proposals
            for idx in range(pad_num):
                rand_idx = np.random.randint(0, num_proposals)
                boxes[i] =  np.row_stack((boxes[i], boxes[i][rand_idx]))
                scores[i] =  np.row_stack((scores[i], scores[i][rand_idx]))
        #print ("boxes[%s] shape:%s"%(i, str(boxes[i].shape)))
        #print ("scores[%s] shape:%s"%(i, str(scores[i].shape)))

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
        #np.save("scores", scores)
        #np.save("boxes", boxes)
        #print ("boxes shape:%s"%str(boxes.shape))
        #print ("scores shape:%s"%str(scores.shape))
        #print ("data_batch.pad:%s"%data_batch.pad)
        t2 = time.time() - t
        t = time.time()

        for ii in range(scores.shape[0] - data_batch.pad):
            # assemble proposals
            #print ("boxes[%s] shape:%s"%(ii, str(boxes[ii].shape)))
            #print ("scores[%s] shape:%s"%(ii, str(scores[ii].shape)))
            #if boxes[ii].shape != scores[ii].shape:
            #    continue
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
        print('generating %d/%d' % (i + 1, imdb.num_images) + ' proposal %d' % (dets.shape[0]) +
              ' data %.4fs net %.4fs post %.4fs' % (t1, t2, t3))

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    return imdb_boxes, original_boxes


def im_detect_mask(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)
    data_dict = dict(zip(data_names, data_batch.data))
    print(output.keys())
    if config.TEST.HAS_RPN:
        # pred_boxes = output['mask_roi_pred_boxes'].asnumpy()
        # scores = output['mask_roi_score'].asnumpy()
        pred_boxes = output['bbox_pred_reshape_output'].asnumpy()
        scores = output['cls_prob_reshape_output'].asnumpy()
        mask_output = output['mask_prob_output'].asnumpy()
    else:
        raise NotImplementedError
    # we used scaled image & roi to train, so it is necessary to transform them back
    if isinstance(scale, float) or isinstance(scale, int):
        pred_boxes = pred_boxes / scale
    elif isinstance(scale, np.ndarray):
        pred_boxes = pred_boxes / scale[:, None, None]

    return scores, pred_boxes, data_dict


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

    img_ind = 0
    t = time.time()
    SCORE_THRESH = 0.05
    for im_infos, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = im_infos[:, 2]
        scores, boxes, data_dict = im_detect_mask(predictor, data_batch, data_names, scales)

        t2 = time.time() - t
        t = time.time()

        for i in range(data_batch.data[0].shape[0] - data_batch.pad):
            score, box, im_info = scores[i], boxes[i], im_infos[i:i + 1]
            roi_rec = roidb[img_ind]
            label = np.argmax(score, axis=1)
            label = label[:, np.newaxis]

            for j in range(1, num_classes):
                inds = np.where(score[:, j] > SCORE_THRESH)[0]
                score_j = score[inds, j]
                box_j = box[inds, j * 4:(j + 1) * 4]
                dets_j = np.hstack((box_j, score_j[:, np.newaxis])).astype(np.float32, copy=False)
                #nms
                keep = nms(dets_j)
                nms_dets = dets_j[keep, :]
                all_boxes[j][img_ind] = nms_dets

            """for cls_ind in range(num_classes):
                cls_boxes = box[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = score[:, cls_ind, np.newaxis]
                keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
                dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = nms(dets)
                all_boxes[cls_ind][img_ind] = dets[keep, :]
            """

            # the first class is empty because it is background
            boxes_this_image = [[]] + [all_boxes[cls_ind][img_ind] for cls_ind in range(1, num_classes)]

            results_list.append({'image': roi_rec['image'],
                                 'im_info': im_info,
                                 'boxes': boxes_this_image})
            t3 = time.time() - t
            t = time.time()
            print('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(img_ind + 1, num_images, t1, t2, t3))
            img_ind += 1

    results_pack = {'all_boxes': all_boxes,
                    'results_list': results_list}

    imdb.evaluate_mask(results_pack)


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
        print(mask_output.shape, scores.shape, boxes.shape)
        scores, boxes = scores[0], boxes[0]

        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        all_masks = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]

        for cls_ind in range(num_classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            # print cls_scores.shape, label.shape
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            # print dets.shape, cls_masks.shape
            all_boxes[cls_ind] = dets[keep, :]
            all_masks[cls_ind] = cls_masks[keep, :, :]

        # the first class is empty because it is background
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, num_classes)]
        masks_this_image = [[]] + [all_masks[j] for j in range(1, num_classes)]
        filename = roi_rec['image'].split("/")[-1]
        filename = result_path + '/' + filename.replace('.png', '') + '.jpg'
        data_dict = dict(zip(data_names, data_batch.data))
        draw_detection_mask(data_dict['data'], boxes_this_image, masks_this_image, 1.0, filename, imdb.classes)
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
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


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
        for i in range(len(dets)):
            bbox = dets[i, :4] * scale
            score = dets[i, -1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            mask = masks[i, :, :]
            if cfg.MASKFCN.ON:
                scale_factor = get_scale_factor(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
                mask_box = [coord / int(scale_factor) for coord in bbox]
                mask = mask[0:mask_box[3] - mask_box[1], 0:mask_box[2] - mask_box[0]]
                mask = cv2.resize(mask, dsize=None, fx=int(scale_factor), fy=int(scale_factor))
                bbox = [coord * int(scale_factor) for coord in mask_box]
            else:
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
    class_names = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'mcycle', 'bicycle')
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array.asnumpy(), config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = boxes_this_image[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    print(filename)
    cv2.imwrite(filename, im)
