"""
Debug Op
"""
from __future__ import print_function

import time
import uuid
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ..core.helper import get_long_side, get_scale_factor


class DebugOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(DebugOperator, self).__init__()
        self.pos = kwargs.get("pos", None)
        self.num_args = int(kwargs.get("num_args", 1))

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()
        print("debugging: %s" % self.pos)
        if self.pos == "roi_feat":
            plt.imshow(data[0, 0])
            plt.show()
            print(np.array2string(data[0, 0, ::3, ::3], precision=2))
        elif self.pos == "mask_target":
            label = in_data[1].asnumpy()
            if (data == 0).all():
                print("all 0 in mask, this is weird")
            plt.imshow(data[0, int(label[0])])
            plt.show()
            print(np.array2string(data[0, int(label[0]), ::7, ::7], precision=2))
        elif self.pos == "rois_lvl":
            print(np.array2string(data[0], precision=2))
        elif self.pos == "feat_lvl":
            plt.imshow(data[0, 0])
            plt.show()
            print(np.array2string(data[0, 0, ::4, ::4], precision=2))
        elif self.pos == "viz":
            rois = in_data[1].asnumpy()
            im = np.transpose(in_data[0].astype(np.uint8).asnumpy()[0], (1, 2, 0))[:, :, ::-1].copy()
            print(rois[:10])
            for roi in rois[:10]:
                roi = list(map(int, roi))
                cv2.rectangle(im, (roi[1], roi[2]), (roi[3], roi[4]), color=(0, 0, 255), thickness=2)
            cv2.imshow("viz", im)
            cv2.waitKey()
        elif self.pos == "seg_viz":
            im = in_data[0].asnumpy()[0][0]
            seg = in_data[1].asnumpy()[0]

            uid = uuid.uuid4()
            cv2.imwrite("visual/%s.jpg" % uid, im)
            cv2.imwrite("visual/%s.png" % uid, seg)
        elif self.pos == "mask_validate":
            im = in_data[0].asnumpy()[0]
            im = np.transpose(im.astype(np.uint8), (1, 2, 0))[:, :, ::-1].copy()
            rois = in_data[1].asnumpy()
            masks = in_data[2].asnumpy()
            labels = in_data[3].asnumpy()

            img = im
            for i in range(5):
                roi = rois[i:i+1]
                scale = get_scale_factor(get_long_side(roi[0]))

                im = img
                im = cv2.resize(im, dsize=None, fx=1.0 / scale, fy=1.0 / scale)
                im = np.transpose(im, (2, 0, 1))
                im = np.expand_dims(im, axis=0)
                im = mx.nd.array(im, dtype=np.float32)
                roi = mx.nd.array(roi)
                stride = scale #* 4
                im_roi = mx.nd.contrib.ROICrop(data=im, rois=roi, pooled_size=(112, 112), spatial_scale=1.0 / stride)

                im_roi = im_roi.asnumpy()[0]
                im_roi = np.transpose(im_roi.astype(np.uint8), (1, 2, 0))[:, :, ::-1].copy()
                # im_roi = cv2.resize(im_roi, dsize=(28, 28))
                # im_roi = cv2.resize(im_roi, dsize=None, fx=4.0, fy=4.0)
                mask = masks[i, int(labels[i])]
                mask = ((mask + 1) / 2.0 * 255).astype(np.uint8)
                # print mask
                uid = uuid.uuid4()
                cv2.imwrite("visual/scale%d-%s.jpg" % (int(scale), uid), im_roi)
                cv2.imwrite("visual/scale%d-%s.png" % (int(scale), uid), mask)
        elif self.pos == "demo_rcnn_proposal":
            im = in_data[0].asnumpy()[0]
            im = np.transpose(im, (1, 2, 0)).copy()
            rois = in_data[1].asnumpy()
            masks = in_data[2].asnumpy()
            labels = in_data[3].asnumpy().astype(np.int32)

            for i in range(3):
                roi = rois[i][1:]
                print("roi: {}".format(roi))
                roi = list(map(int, roi))
                cv2.rectangle(im, (roi[0], roi[1]), (roi[2], roi[3]), color=(0, 0, 255), thickness=2)
                cls = labels[i]
                assert cls != 0, "top roi are not foreground"
                mask = masks[i][cls]
                resized_mask = cv2.resize(mask, dsize=(roi[2] - roi[0], roi[3] - roi[1]))  # dsize=(x, y)
                resized_mask = np.array(resized_mask > 0.5)
                resized_mask = resized_mask * np.random.randint(80, 100)
                im[roi[1]:roi[3], roi[0]:roi[2], i] += resized_mask
            im[im > 255] = 255
            im = im.astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite("visual/%s.jpg" % uuid.uuid4(), im)
        else:
            raise ValueError("unknown debug pos: %s" % self.pos)
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("Debug")
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(DebugProp, self).__init__(need_top_grad=True)
        self._kwargs = kwargs

    def list_arguments(self):
        inputs = ['data']
        num_args = int(self._kwargs.get("num_args", 1))
        if num_args > 1:
            for i in range(1, num_args):
                inputs.append("data%d" % i)
        return inputs

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, in_shape[:1]

    def create_operator(self, ctx, shapes, dtypes):
        return DebugOperator(**self._kwargs)
