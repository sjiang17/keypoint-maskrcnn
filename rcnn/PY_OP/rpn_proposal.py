"""
rpn_proposal operator
"""

import mxnet as mx
import numpy as np
from rcnn.config import config

DEBUG = False

class RPNProposalOperator(mx.operator.CustomOp):
    def __init__(self):
        super(RPNProposalOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        """for i, stride in enumerate(self._feat_stride_fpn):
            fpn_feat_pyramid.update({'stride%s' % stride: in_data[i]})
        """

        rois = in_data[-1]
        #np.save("rois", rois.asnumpy())
        #rois_np = rois.asnumpy()
        #rois = mx.nd.array(rois_np)
        roi_scores = in_data[-2]
        roi_concat = mx.ndarray.concatenate([rois, roi_scores], 1).asnumpy()
        roi_concat = mx.nd.array(roi_concat)
        roi_sort = roi_concat[mx.ndarray.argsort(roi_concat[:, 5], is_ascend=0)]
        #np.save("roi_scores", roi_scores.asnumpy())
        #np.save("roi_sort", roi_sort.asnumpy())
        #np.save("roi_top", roi_sort[:1000,:5].asnumpy())
        self.assign(out_data[0], req[0], roi_sort[:config.TEST.RPN_POST_NMS_TOP_N,:5])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("rpn_proposal")
class RPNProposalProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(RPNProposalProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        args_list = []
        """feat_stride_fpn = np.fromstring(self._rcnn_strides[1:-1], dtype=int, sep=',')
        for stride in feat_stride_fpn:
            args_list.append('feat_stride%s' % stride)
        """
        args_list.append('roi_scores')
        args_list.append('rois')
        return args_list

    def list_outputs(self):
        return ['rpn_proposal_roi']

    def infer_shape(self, in_shape):
        out_poo_feat_shape = [1000, 5]
        return in_shape, [out_poo_feat_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return RPNProposalOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
