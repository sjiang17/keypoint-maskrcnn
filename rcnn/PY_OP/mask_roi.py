"""
Make rois for mask branch.
"""

import mxnet as mx
from rcnn.config import config as cfg
from rcnn.processing.bbox_transform import *

bbox_pred = nonlinear_pred


class MaskROIOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(MaskROIOperator, self).__init__()
        self.num_classes = int(kwargs.get("num_classes"))
        self.topk = int(kwargs.get("topk"))
        assert self.num_classes
        assert self.topk

    def forward(self, is_train, req, in_data, out_data, aux):
        """
        The input shape should be (#img, #rois, ...), since multi-image test is supported
        :param is_train:
        :param req:
        :param in_data:
        :param out_data:
        :param aux:
        :return:
        """
        rois = in_data[0].asnumpy().reshape((-1, 5))  # (#img, #roi, 5)
        scores = in_data[3].asnumpy()  # (#img, #roi, #cls)
        num_cls = scores.shape[-1]
        scores = scores.reshape((-1, scores.shape[-1]))
        bbox_deltas = in_data[1].asnumpy()  # (#img, #roi, #cls * 4)
        num_img, num_roi = bbox_deltas.shape[:2]
        bbox_deltas = bbox_deltas.reshape((-1, num_cls * 4))
        if cfg.MASKFCN.RPN_BBOX_FOR_MASK:
            bbox_deltas[:] = 0
        data = in_data[2]  # (#img, #roi, c, h, w)
        im_infos = in_data[4].asnumpy()  # (#img, 3)

        if is_train:
            raise NotImplementedError()
        else:
            # encode -> xyxy
            pred_boxes = bbox_pred(rois[:, 1:], bbox_deltas)
            pred_boxes = clip_boxes(pred_boxes, im_infos[0, :2])
            #print im_infos
            # class-specific preds
            output_boxes = np.zeros(shape=(rois.shape[0], 4), dtype=np.float32)
            labels = np.argmax(scores, axis=1).astype(np.int32)  # (#img * #roi, )
            for i in range(num_img * num_roi):
                cls = labels[i]
                output_boxes[i] = pred_boxes[i, 4 * cls: 4 * (cls + 1)]
            # clip preds to boundary
            # output_boxes = clip_boxes(output_boxes, data.shape[-2:])
            # pad batch inds for ROIXFeat
            batch_inds = np.zeros(shape=output_boxes.shape[:1] + (1, ), dtype=np.float32)
            output_boxes = np.concatenate((batch_inds, output_boxes), axis=1)
            # take topk
            top_inds = np.argsort(np.max(scores, axis=1))[-self.topk:]
            #output_boxes = output_boxes.reshape((num_img, self.topk, 5))
            #pred_boxes = pred_boxes.reshape((num_img, self.topk, num_cls * 4))
            #scores = scores.reshape((num_img, self.topk, num_cls))
            output_boxes = output_boxes[top_inds].reshape((num_img, self.topk, 5))
            pred_boxes = pred_boxes[top_inds].reshape((num_img, self.topk, num_cls * 4))
            scores = scores[top_inds].reshape((num_img, self.topk, num_cls))

            self.assign(out_data[0], req[0], output_boxes)
            self.assign(out_data[1], req[1], pred_boxes)
            self.assign(out_data[2], req[2], scores)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)


@mx.operator.register("MaskROI")
class MaskROIProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(MaskROIProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs

    def list_arguments(self):
        return ['rois', 'bbox_deltas', 'data', 'label', 'im_info']

    def list_outputs(self):
        return ['mask_roi', 'pred_boxes', 'score']

    def infer_shape(self, in_shape):
        topk = int(self._kwargs.get("topk"))
        assert topk

        rois_shape = in_shape[0]
        bbox_deltas_shape = in_shape[1]
        data_shape = in_shape[2]
        label_shape = in_shape[3]
        im_info_shape = in_shape[4]

        num_img = bbox_deltas_shape[0]

        return [rois_shape,
                bbox_deltas_shape,
                data_shape,
                label_shape,
                im_info_shape], \
               [(num_img, topk, rois_shape[-1]),
                (num_img, topk, bbox_deltas_shape[-1]),
                (num_img, topk, label_shape[-1])]

    def create_operator(self, ctx, shapes, dtypes):
        return MaskROIOperator(**self._kwargs)
