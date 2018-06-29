"""
fpn roi pooling
"""

import mxnet as mx
import numpy as np
from rcnn.config import config

DEBUG = False


def _unmap(data, count, inds):
    """" unmap a subset inds of data into original data of size count """
    assert len(inds) == count
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret[inds, :] = data
    return ret


class ROICropFPN(mx.operator.CustomOp):
    def __init__(self, rcnn_strides, pool_h, pool_w):
        super(ROICropFPN, self).__init__()
        self._feat_stride_fpn = np.fromstring(rcnn_strides[1:-1], dtype=int, sep=',')
        self._pool_h = int(pool_h)
        self._pool_w = int(pool_w)

    def forward(self, is_train, req, in_data, out_data, aux):
        fpn_feat_pyramid = {}
        for i, stride in enumerate(self._feat_stride_fpn):
            fpn_feat_pyramid.update({'stride%s' % stride: in_data[i]})

        rois = in_data[-1]
        rois_cpu = rois.asnumpy()
        num_rois = rois.shape[0]
        rois_long_side = np.max(
            np.concatenate([rois_cpu[:, 3:4] - rois_cpu[:, 1:2], rois_cpu[:, 4:5] - rois_cpu[:, 2:3]], axis=1), axis=1)

        if DEBUG:
            print 'rois_area shape:', rois_long_side.shape

        feat_dict = {}
        for stride in self._feat_stride_fpn:
            feat_dict.update({'stride%s' % stride: fpn_feat_pyramid['stride%s' % stride]})

        long_side_threshold = {'stride32': [np.inf, 448],
                               'stride16': [448, 224],
                               'stride8': [224, 112],
                               'stride4': [112, 0]}

        # Assign to levels
        roi_pool_list = []
        index_list = []
        for s in self._feat_stride_fpn:
            thresh = long_side_threshold['stride%s' % s]
            index = np.where(np.logical_and(thresh[1] <= rois_long_side, rois_long_side < thresh[0]))[0]
            if DEBUG:
                print "stride: %s, num rois: %d" % (s, len(index))

            if len(index) > 0:
                index = mx.nd.array(index, rois.context)
                if DEBUG:
                    print 'Context:'
                    print 'feat:', feat_dict['stride%s' % s].context
                    print 'rois:', rois.context
                    print 'index:', index.context
                _rois = mx.nd.take(rois, index)
                roi_pool = mx.nd.contrib.ROICrop(feat_dict['stride%s' % s], _rois, (self._pool_h, self._pool_w),
                                                 1.0 / float(s))
                roi_pool_list.append(roi_pool)
                index_list.append(index.asnumpy())
        fpn_roi_pool = mx.nd.concatenate(roi_pool_list, axis=0)
        index_concat = np.concatenate(index_list)
        fpn_roi_pool = _unmap(fpn_roi_pool.asnumpy(), num_rois, index_concat.astype(int))
        self.assign(out_data[0], req[0], fpn_roi_pool)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError


@mx.operator.register("ROICropFPN")
class ROICropFPNProp(mx.operator.CustomOpProp):
    def __init__(self, rcnn_strides='(32,16,8,4)', pool_h='30', pool_w='30'):
        super(ROICropFPNProp, self).__init__(need_top_grad=True)
        self._pool_h = int(pool_h)
        self._pool_w = int(pool_w)
        self._rcnn_strides = rcnn_strides

    def list_arguments(self):
        args_list = []
        feat_stride_fpn = np.fromstring(self._rcnn_strides[1:-1], dtype=int, sep=',')
        for stride in feat_stride_fpn:
            args_list.append('feat_stride%s' % stride)
        args_list.append('rois')
        return args_list

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        out_poo_feat_shape = [in_shape[-1][0], in_shape[0][1], self._pool_h, self._pool_w]
        return in_shape, [out_poo_feat_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return ROICropFPN(self._rcnn_strides, self._pool_h, self._pool_w)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
