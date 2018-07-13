import mxnet as mx
import numpy as np
from rcnn.config import config


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    return pred, label


def get_rcnn_fpn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['label', 'bbox_target', 'bbox_weight']
    return pred, label


def get_maskrcnn_fpn_name():
    rcnn_pred, rcnn_label = get_rcnn_fpn_names()
    pred = rcnn_pred + ['keypoint_prob']
    label = rcnn_label + ['keypoint_target']
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        label = labels[self.label.index("label")].asnumpy().astype("int32").flatten()

        pred = preds[self.pred.index('rcnn_cls_prob')]
        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')

        valid_index = np.where(label != -1)
        label = label[valid_index]
        pred_label = pred_label[valid_index]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class MaskLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskLogLossMetric, self).__init__('MaskLogLoss')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        # reshape and concat
        mask_prob = preds[self.pred.index('mask_prob')].asnumpy()  # (n_rois, c, h, w)
        mr = mask_prob.shape[-1]  # mask resolution
        label = labels[self.label.index('label')].asnumpy().reshape((-1,)).astype("int32")
        mask_target = labels[self.label.index('mask_target')].asnumpy().reshape((-1, config.NUM_CLASSES, mr, mr))
        mask_weight = labels[self.label.index('mask_weight')].asnumpy().reshape((-1, config.NUM_CLASSES, 1, 1))

        real_inds = np.where(label != 0)[0]
        n_rois = real_inds.shape[0]
        mask_prob = mask_prob[real_inds, label[real_inds]]
        mask_target = mask_target[real_inds, label[real_inds]]
        mask_weight = mask_weight[real_inds, label[real_inds]]
        l = mask_weight * mask_target * np.log(mask_prob + 1e-14) + mask_weight * (1 - mask_target) * np.log(
            1 - mask_prob + 1e-14)
        self.sum_metric += -np.sum(l)
        self.num_inst += mask_prob.shape[-1] * mask_prob.shape[-2] * n_rois


class MaskAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskAccMetric, self).__init__('MaskACC')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        # reshape and concat
        mask_prob = preds[self.pred.index('mask_prob')].asnumpy()  # (n_rois, c, h, w)
        mr = mask_prob.shape[-1]  # mask resolution
        label = labels[self.label.index('label')].asnumpy().reshape((-1,)).astype("int32")
        # print('label:', label)
        mask_target = labels[self.label.index('mask_target')].asnumpy().reshape((-1, config.NUM_CLASSES, mr, mr))
        # print('index:', self.label.index('mask_weight'))
        mask_weight = labels[self.label.index('mask_weight')].asnumpy().reshape((-1, config.NUM_CLASSES, 1, 1))

        real_inds = np.where(label != 0)[0]  # foreground mask only
        n_rois = real_inds.shape[0]
        mask_prob = mask_prob[real_inds, label[real_inds]]
        mask_target = mask_target[real_inds, label[real_inds]]
        mask_weight = mask_weight[real_inds, label[real_inds]]
        idx = np.where(np.logical_and(mask_prob > 0.5, mask_weight == 1))
        mask_pred = np.zeros_like(mask_prob)
        mask_pred[idx] = 1
        self.sum_metric += np.sum(mask_target == mask_pred)
        self.num_inst += mask_prob.shape[-1] * mask_prob.shape[-2] * n_rois


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        label = labels[self.label.index("label")].asnumpy().astype("int32").flatten()

        pred = preds[self.pred.index('rcnn_cls_prob')]
        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)

        cls = pred[np.arange(label.shape[0]), label]

        index = np.where(label != -1)
        cls = cls[index]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += index[0].shape[0]


class KeypointLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(KeypointLossMetric, self).__init__('KeypointLoss')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        label = labels[self.label.index('keypoint_target')].asnumpy().astype("int32")
        # print('label shape', label.shape)
        pred = preds[self.pred.index('keypoint_prob')].asnumpy()
        # print('pred shape', pred.shape)
        # pred = pred.reshape((-1, 56*56))
        cls = pred[np.arange(label.shape[0]), label]
        index = np.where(label != -1)
        cls = cls[index]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += index[0].shape[0]


class RPNRegLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        name = 'RPNL1Loss'
        super(RPNRegLossMetric, self).__init__(name)
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNRegLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        name = 'RCNNL1Loss'
        super(RCNNRegLossMetric, self).__init__(name)
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        label = labels[self.label.index("label")].asnumpy().astype("int32").flatten()

        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        last_dim = bbox_loss.shape[-1]
        bbox_loss = bbox_loss.reshape((-1, last_dim))

        # calculate num_inst
        keep_inds = np.where((label != 0) & (label != -1))[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))


class AccWithIgnoreMetric(mx.metric.EvalMetric):
    def __init__(self, output_names, label_names, ignore_label=255, name='AccWithIgnore'):
        super(AccWithIgnoreMetric, self).__init__(name=name, output_names=output_names, label_names=label_names)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat) - (label.flat == self.ignore_label).sum()


class IoUMetric(mx.metric.EvalMetric):
    def __init__(self, ignore_label, label_num, name='IoU'):
        self._ignore_label = ignore_label
        self._label_num = label_num
        super(IoUMetric, self).__init__(name=name)

    def reset(self):
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            iou = 0
            eps = 1e-6
            # skip_label_num = 0
            for j in range(self._label_num):
                pred_cur = (pred_label.flat == j)
                gt_cur = (label.flat == j)
                tp = np.logical_and(pred_cur, gt_cur).sum()
                denom = np.logical_or(pred_cur, gt_cur).sum() - np.logical_and(pred_cur, label.flat == self._ignore_label).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self._label_num
            self.sum_metric = iou
            self.num_inst = 1
