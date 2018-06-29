import numpy as np
import mxnet as mx
# import cv2
import gc
import time
import logging
from unittests.utils import log_time
from rcnn.io.image import tensor_vstack

logging.basicConfig(level=logging.INFO)


class ROICropOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(ROICropOperator, self).__init__()
        self.seg_code = int(kwargs.get("seg_code", 1000))
        self.num_class = int(kwargs.get("num_class", 9))
        self.grad_scale = float(kwargs.get("grad_scale", 1.0))
        self.in_data_list = []
        self.loss = None

    def forward(self, is_train, req, in_data, out_data, aux):
        forward_start = time.time()

        class_id = [0, 24, 25, 26, 27, 28, 31, 32, 33]

        feat, deconv1_weight, deconv2_weight, conv1_weight, conv2_weight, gt, rois, label = in_data
        if rois.shape[1] == 5:
            rois = rois[:, 1:]

        for blob in [feat, deconv1_weight, deconv2_weight, conv1_weight, conv2_weight]:
            blob.attach_grad()
            self.in_data_list.append(blob)

        with log_time("asnumpy"):
            gt = gt.asnumpy()
            label = label.astype(int).asnumpy()
            rois = rois.asnumpy()
            rois_stride4 = np.ceil(rois / 4).astype(int)
            feat_np = feat.asnumpy()

        fg_indexes = np.where(label != 0)[0]
        fg_rois = rois[fg_indexes]
        fg_roi_areas = (fg_rois[:, 2] - fg_rois[:, 0]) * (fg_rois[:, 3] - fg_rois[:, 1])
        fg_small_indexes = fg_indexes[np.where(fg_roi_areas < 5000)[0]]
        fg_middle_indexes = fg_indexes[np.where(np.logical_and(fg_roi_areas >= 5000, fg_roi_areas < 30000))]
        fg_large_indexes = fg_indexes[np.where(fg_roi_areas >= 30000)[0]]

        self.fg_small_rois = rois_stride4[fg_small_indexes]
        self.fg_middle_rois = rois_stride4[fg_middle_indexes]

        loss = mx.nd.array([0.0])
        if len(self.fg_small_rois) > 0:
            with log_time("prepare small"):
                feat_tensor_list = []
                for roi in rois_stride4[fg_small_indexes]:
                    x0, y0, x1, y1 = roi
                    feat_tensor_list.append(feat_np[:, :, y0:y1, x0:x1])
                pad_small_feat = tensor_vstack(feat_tensor_list)
                pad_small_feat = mx.nd.array(pad_small_feat, ctx=feat.context)
                pad_small_feat.attach_grad()
                self.in_data_list.append(pad_small_feat)

                gt_tensor_list = []
                for i in fg_small_indexes:
                    x0, y0, x1, y1 = rois[i]
                    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
                    crop_gt = gt[0, y0:y1, x0:x1].copy()
                    h, w = crop_gt.shape
                    if h > w:
                        rand_points = [np.arange(h), np.random.randint(0, w, (h,))]
                    else:
                        rand_points = [np.random.randint(0, h, (w,)), np.arange(w)]
                    points = crop_gt[rand_points]
                    ids, counts = np.unique(points, return_counts=True)
                    indexes_sort_counts = np.argsort(counts)
                    mask = None
                    for idx in indexes_sort_counts:
                        if np.floor(ids[idx] / self.seg_code) == class_id[int(label[i])]:
                            crop_gt[crop_gt != ids[idx]] = 0.
                            crop_gt[crop_gt == ids[idx]] = 1.
                            mask = crop_gt
                            break
                    if mask is None:
                        mask = np.zeros_like(crop_gt)
                    tmp = np.zeros(shape=(1, self.num_class) + mask.shape[-2:])
                    tmp[0, label[i]] = mask
                    gt_tensor_list.append(tmp)
                pad_small_gt = tensor_vstack(gt_tensor_list)
                pad_small_gt = mx.nd.array(pad_small_gt, ctx=feat.context)

            with log_time("deconv small"):
                with mx.autograd.record():
                    deconv1 = mx.nd.Deconvolution(pad_small_feat, weight=deconv1_weight, num_filter=32, kernel=(2, 2),
                                                  stride=(2, 2), no_bias=True, cudnn_tune="off")
                    conv1 = mx.nd.Convolution(deconv1, weight=conv1_weight, num_filter=32, kernel=(3, 3), pad=(1, 1),
                                              no_bias=True, cudnn_tune="off")
                    relu1 = mx.nd.Activation(conv1, act_type="relu")
                    deconv2 = mx.nd.Deconvolution(relu1, weight=deconv2_weight, num_filter=32, kernel=(2, 2),
                                                  stride=(2, 2), pad=(2, 2), no_bias=True, cudnn_tune="off")
                    fullconv = mx.nd.Convolution(deconv2, weight=conv2_weight, num_filter=self.num_class,
                                                 kernel=(1, 1), no_bias=True, cudnn_tune="off")
                    prob = mx.nd.Activation(fullconv, act_type="sigmoid")
                    crop_mask = mx.nd.Crop(pad_small_gt, prob)
                    loss = loss + mx.nd.mean(mx.nd.square(crop_mask - prob))

        if len(self.fg_middle_rois) > 0:
            with log_time("prepare middle"):
                feat_tensor_list = []
                for roi in rois_stride4[fg_middle_indexes]:
                    x0, y0, x1, y1 = roi
                    feat_tensor_list.append(feat_np[:, :, y0:y1, x0:x1])
                pad_middle_feat = tensor_vstack(feat_tensor_list)
                pad_middle_feat = mx.nd.array(pad_middle_feat, ctx=feat.context)
                pad_middle_feat.attach_grad()
                self.in_data_list.append(pad_middle_feat)

                gt_tensor_list = []
                for i in fg_middle_indexes:
                    x0, y0, x1, y1 = rois[i]
                    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
                    crop_gt = gt[0, y0:y1, x0:x1].copy()
                    h, w = crop_gt.shape
                    if h > w:
                        rand_points = [np.arange(h), np.random.randint(0, w, (h,))]
                    else:
                        rand_points = [np.random.randint(0, h, (w,)), np.arange(w)]
                    points = crop_gt[rand_points]
                    ids, counts = np.unique(points, return_counts=True)
                    indexes_sort_counts = np.argsort(counts)
                    mask = None
                    for idx in indexes_sort_counts:
                        if np.floor(ids[idx] / self.seg_code) == class_id[int(label[i])]:
                            crop_gt[crop_gt != ids[idx]] = 0.
                            crop_gt[crop_gt == ids[idx]] = 1.
                            mask = crop_gt
                            break
                    if mask is None:
                        mask = np.zeros_like(crop_gt)
                    tmp = np.zeros(shape=(1, self.num_class) + mask.shape[-2:])
                    tmp[0, label[i]] = mask
                    gt_tensor_list.append(tmp)
                pad_middle_gt = tensor_vstack(gt_tensor_list)
                pad_middle_gt = mx.nd.array(pad_middle_gt, ctx=feat.context)

            with log_time("deconv middle"):
                with mx.autograd.record():
                    deconv1 = mx.nd.Deconvolution(pad_middle_feat, weight=deconv1_weight, num_filter=32, kernel=(2, 2),
                                                  stride=(2, 2), no_bias=True, cudnn_tune="off")
                    conv1 = mx.nd.Convolution(deconv1, weight=conv1_weight, num_filter=32, kernel=(3, 3), pad=(1, 1),
                                              no_bias=True, cudnn_tune="off")
                    relu1 = mx.nd.Activation(conv1, act_type="relu")
                    deconv2 = mx.nd.Deconvolution(relu1, weight=deconv2_weight, num_filter=32, kernel=(2, 2),
                                                  stride=(2, 2), pad=(2, 2), no_bias=True, cudnn_tune="off")
                    fullconv = mx.nd.Convolution(deconv2, weight=conv2_weight, num_filter=self.num_class,
                                                 kernel=(1, 1), no_bias=True, cudnn_tune="off")
                    prob = mx.nd.Activation(fullconv, act_type="sigmoid")
                    crop_mask = mx.nd.Crop(pad_middle_gt, prob)
                    loss = loss + mx.nd.mean(mx.nd.square(crop_mask - prob))

        with mx.autograd.record():
            with log_time("large rois"):
                for i in fg_large_indexes.tolist():
                    # crop gt
                    roi = np.ceil(rois[i]).astype(int)  # slice is left inclusive and right exclusive
                    x0, y0, x1, y1 = roi
                    if x0 >= x1 or y0 >= y1:
                        continue

                    crop_gt = gt[0, y0:y1, x0:x1].copy()
                    h, w = crop_gt.shape
                    if h > w:
                        rand_points = [np.arange(h), np.random.randint(0, w, (h,))]
                    else:
                        rand_points = [np.random.randint(0, h, (w,)), np.arange(w)]
                    points = crop_gt[rand_points]
                    ids, counts = np.unique(points, return_counts=True)
                    indexes_sort_counts = np.argsort(counts)

                    # gt_copy = gt[0].copy()
                    # cv2.rectangle(gt_copy, (x0, y0), (x1, y1), color=255*255, thickness=2)
                    # cv2.imshow("", gt_copy.astype(np.uint16))
                    # cv2.waitKey()
                    mask = None
                    for idx in indexes_sort_counts[::-1]:
                        if np.floor(ids[idx] / self.seg_code) == class_id[label[i]]:
                            crop_gt[crop_gt != ids[idx]] = 0
                            crop_gt[crop_gt == ids[idx]] = 1
                            mask = crop_gt
                            # cv2.imshow("", mask)
                            # cv2.waitKey()
                            break
                    if mask is None:
                        mask = np.zeros_like(crop_gt)
                    tmp = np.zeros(shape=(1, self.num_class) + mask.shape[-2:])
                    tmp[0, label[i]] = mask
                    mask = mx.nd.array(tmp, feat.context)

                    roi_stride4 = np.ceil(rois[i] / 4).astype(int)
                    x0, y0, x1, y1 = roi_stride4
                    crop_feat = mx.nd.slice(feat, begin=(0, 0, y0, x0), end=(1, 128, y1, x1))
                    deconv1 = mx.nd.Deconvolution(crop_feat, weight=deconv1_weight, num_filter=32, kernel=(2, 2),
                                                  stride=(2, 2), no_bias=True, cudnn_tune="off")
                    conv1 = mx.nd.Convolution(deconv1, weight=conv1_weight, num_filter=32, kernel=(3, 3), pad=(1, 1),
                                              no_bias=True, cudnn_tune="off")
                    relu1 = mx.nd.Activation(conv1, act_type="relu")
                    deconv2 = mx.nd.Deconvolution(relu1, weight=deconv2_weight, num_filter=32, kernel=(2, 2),
                                                  stride=(2, 2), pad=(2, 2), no_bias=True, cudnn_tune="off")
                    fullconv = mx.nd.Convolution(deconv2, weight=conv2_weight, num_filter=self.num_class,
                                                 kernel=(1, 1), no_bias=True, cudnn_tune="off")
                    prob = mx.nd.Activation(fullconv, act_type="sigmoid")
                    crop_mask = mx.nd.Crop(mask, prob)
                    loss = loss + mx.nd.mean(mx.nd.square(crop_mask - prob))

                loss = loss / len(fg_indexes)
                loss = loss * self.grad_scale

        logging.debug("loss: %.1f" % loss.asscalar())
        self.loss = loss
        self.assign(out_data[0], req[0], loss)

        forward_stop = time.time()
        logging.debug("forward_consume: %.1fms\n" % ((forward_stop - forward_start) * 1000))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        with mx.autograd.record():
            self.loss.backward()

        self.in_data_list[0].grad.wait_to_read()

        with log_time("backward addup"):
            large_rois_grad = self.in_data_list[0].grad.asnumpy()

            if len(self.fg_small_rois) > 0:
                small_rois_grad = self.in_data_list[5].grad.asnumpy()
                for idx, roi in enumerate(self.fg_small_rois):
                    x0, y0, x1, y1 = roi
                    large_rois_grad[0, :, y0:y1, x0:x1] += small_rois_grad[idx, :, 0:(y1-y0), 0:(x1-x0)]

            if len(self.fg_middle_rois) > 0:
                middle_rois_grad = self.in_data_list[-1].grad.asnumpy()
                for idx, roi in enumerate(self.fg_middle_rois):
                    x0, y0, x1, y1 = roi
                    large_rois_grad[0, :, y0:y1, x0:x1] += middle_rois_grad[idx, :, 0:(y1 - y0), 0:(x1 - x0)]

        self.assign(in_grad[0], req[0], large_rois_grad)
        self.assign(in_grad[1], req[1], self.in_data_list[1].grad)
        self.assign(in_grad[2], req[2], self.in_data_list[2].grad)
        self.assign(in_grad[3], req[3], self.in_data_list[3].grad)
        self.assign(in_grad[4], req[4], self.in_data_list[4].grad)

        self.in_data_list = []  # clear temporary state
        gc.collect()


@mx.operator.register("ROICrop")
class ROICropProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(ROICropProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs

    def list_arguments(self):
        return ['data', 'deconv1_weight', 'deconv2_weight', 'conv1_weight', 'conv2_weight', 'gt', 'rois', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        feat_shape = in_shape[0]
        gt_shape = in_shape[5]
        rois_shape = in_shape[6]
        label_shape = in_shape[7]
        assert rois_shape[0] == label_shape[0], "number of rois and labels mismatch"

        num_class = int(self._kwargs.get("num_class"))
        return [feat_shape, (128, 32, 2, 2), (32, 32, 2, 2), (32, 32, 3, 3),
                (num_class, 32, 1, 1), gt_shape, rois_shape, label_shape], [(1, )], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        gt_type = in_type[5]
        rois_type = in_type[6]
        label_type = in_type[7]
        return [dtype, dtype, dtype, dtype, dtype, gt_type, rois_type, label_type], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return ROICropOperator(**self._kwargs)
