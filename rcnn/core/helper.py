from __future__ import print_function

import numpy as np

from rcnn.processing.bbox_transform import bbox_overlaps


def get_long_side(rois):
    if isinstance(rois, list):
        assert len(rois) == 4 or len(rois) == 5, "roi should be 4/5 tuple, w.o./w batch index"
        if len(rois) == 5:
            rois = rois[1:]
    elif isinstance(rois, np.ndarray):
        assert rois.shape[0] == 4 or rois.shape[0] == 5, "roi should be 4/5 tuple, w.o./w batch index"
        if rois.shape[0] == 5:
            rois = rois[1:]
    else:
        raise ValueError("unknown roi type: %s" % type(rois))
    return max(rois[2] - rois[0], rois[3] - rois[1])


def get_scale_factor(px):
    assert px > 0, "long side of roi should be positive"
    if px <= 112:
        return 1.0
    return 2 ** np.ceil(np.log2(px / 112.0))


def test_single_class_ap():
    gt_boxes = np.array([[0, 0, 100, 100]])
    pred_boxes = np.array([[10, 10, 90, 90], [20, 20, 80, 80], [30, 30, 70, 70], [40, 40, 60, 60]])
    scores = np.array([0.6, 0.1, 0.3, 0.9])
    gt_boxes, pred_boxes, scores = [gt_boxes], [pred_boxes], [scores]

    print(single_class_ap(gt_boxes, pred_boxes, scores, 0.6))


def single_class_ap(gt_boxes, pred_boxes, scores, threshold):
    """
    single_class_ap
        - single_image_single_class_confusion_matrix
        - ap_from_precision_and_recall
    :param gt_boxes: list of (#gt, 4)
    :param pred_boxes: list of (#box, 4)
    :param scores: list of (#box, )
    :param threshold: float, IoU threshold
    :return: ap
    """
    num_img = len(gt_boxes)
    tp_list, fp_list, n_gt = [], [], 0
    for i in range(num_img):
        n_gt += gt_boxes[i].shape[0]
        tp, fp = single_image_single_class_confusion_matrix(gt_boxes[i], pred_boxes[i], scores[i], threshold)
        tp_list.append(tp)
        fp_list.append(fp)

    tp = np.concatenate(tp_list)
    fp = np.concatenate(fp_list)
    scores = np.concatenate(scores)
    sort_inds = np.argsort(-scores)
    tp = tp[sort_inds]
    fp = fp[sort_inds]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / (n_gt + 1e-12)
    precision = tp / (tp + fp)

    return ap_from_precision_and_recall(precision, recall)


def ap_from_precision_and_recall(prec, rec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def single_image_single_class_confusion_matrix(gt_boxes, pred_boxes, scores, threshold):
    """
    :param gt_boxes: (#gt, 4)
    :param pred_boxes: (#box, 4)
    :param scores: (#box, )
    :param threshold: float, IoU threshold
    :return: tp, fp, fn
    """
    if pred_boxes.size == 0:
        return np.zeros(shape=(0, )), np.zeros(shape=(0, ))
    if gt_boxes.size == 0:
        return np.zeros(shape=(pred_boxes.shape[0], )), np.ones(shape=(pred_boxes.shape[0], ))

    ious = bbox_overlaps(pred_boxes.astype(np.float32, copy=False),
                         gt_boxes.astype(np.float32, copy=False))  # (#box, #gt)
    max_overlap_for_boxes = np.max(ious, axis=1)
    gt_for_boxes = np.argmax(ious, axis=1)
    gt_detected = [False] * ious.shape[1]
    tp, fp = np.zeros(shape=(pred_boxes.shape[0], )), np.zeros(shape=(pred_boxes.shape[0], ))
    for ind in np.argsort(-scores):
        overlap = max_overlap_for_boxes[ind]
        gt_ind = gt_for_boxes[ind]
        if overlap >= threshold:
            if not gt_detected[gt_ind]:
                tp[ind] = 1
                gt_detected[gt_ind] = True
            else:
                fp[ind] = 1
        else:
            fp[ind] = 1
    return tp, fp


if __name__ == "__main__":
    test_single_class_ap()