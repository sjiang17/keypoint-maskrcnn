"""
Renormalize Global Average Pooling Op
"""

import mxnet as mx
import numpy as np

from ..core.helper import get_scale_factor
from ..config import config as cfg


class RenormGAPOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(RenormGAPOperator, self).__init__()
        self.pos = kwargs.get("pos", None)
        self.num_args = int(kwargs.get("num_args", 1))

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()[:]  # (n, 5), (batch_idx, x1, y1, x2, y2)
        renorm = np.zeros(shape=(rois.shape[0], 1, 1, 1), dtype=np.float32)
        for n, roi in enumerate(rois):
            long_side = max(roi[4] - roi[2], roi[3] - roi[1])
            scale_factor = 4 * get_scale_factor(long_side)
            roi /= scale_factor
            renorm[n] = 784 / (int(roi[4] - roi[2] + 1) * int(roi[3] - roi[1] + 1))
        self.assign(out_data[0], req[0], renorm)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError()


@mx.operator.register("RenormGAP")
class RenormGAPProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(RenormGAPProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        return in_shape, [(batch_size, 1, 1, 1)]

    def create_operator(self, ctx, shapes, dtypes):
        return RenormGAPOperator(**self._kwargs)
