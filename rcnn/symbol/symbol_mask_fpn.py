import mxnet as mx
import sys
sys.path.append('/mnt/truenas/scratch/siyu/maskrcnn')
import rcnn
from rcnn.config import config
from rcnn.config import config as cfg
from rcnn.PY_OP import fpn_roi_pooling, fpn_roi_crop, proposal_fpn, mask_roi, debug, renormalize_global_avg_pool,rpn_proposal

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps['50']
filter_list = [256, 512, 1024, 2048]


def residual_unit(data, num_filter, stride, dim_match, name, dilate=(1, 1)):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                               dilate=dilate, pad=dilate, no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')

    if cfg.MEMONGER:
        sum._set_attr(mirror_stage='True')

    return sum


def get_resnet_conv(data):
    # res1, stride 4
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2, stride 4
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i)
    conv_c2 = unit

    # res3, stride 8
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i)
    conv_c3 = unit

    # res4, stride 16
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    conv_c4 = unit

    # res5, stride 32
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i)
    conv_c5 = unit

    conv_feat = [conv_c5, conv_c4, conv_c3, conv_c2]
    return conv_feat



def get_resnet_conv_down(conv_feat):
    # C5 to P5, 1x1 dimension reduction to 256
    p5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")
    p5_conv = mx.symbol.Convolution(data=p5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5_conv")

    # P5 2x upsampling + C4 = P4
    p5_up = mx.symbol.UpSampling(p5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    p4_la = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
    p5_clip = mx.symbol.Crop(*[p5_up, p4_la], name="P4_clip")
    p4 = mx.sym.ElementWiseSum(*[p5_clip, p4_la], name="P4_sum")
    #p4 = p5_clip + p4_la
    p4_conv = mx.symbol.Convolution(data=p4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_conv")

    # P4 2x upsampling + C3 = P3
    p4_up = mx.symbol.UpSampling(p4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    p3_la = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
    p4_clip = mx.symbol.Crop(*[p4_up, p3_la], name="P3_clip")
    p3 = mx.sym.ElementWiseSum(*[p4_clip, p3_la], name="P3_sum")
    #p3 = p4_clip + p3_la
    p3_conv = mx.symbol.Convolution(data=p3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_conv")

    # P3 2x upsampling + C2 = P2
    p3_up = mx.symbol.UpSampling(p3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    p2_la = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
    p3_clip = mx.symbol.Crop(*[p3_up, p2_la], name="P2_clip")
    p2 = mx.sym.ElementWiseSum(*[p3_clip, p2_la], name="P2_sum")
    #p2 = p3_clip + p2_la
    p2_conv = mx.symbol.Convolution(data=p2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_conv")

    # P6 2x subsampling P5
    p6 = mx.symbol.Pooling(data=p5_conv, kernel=(1, 1), stride=(2, 2), pad=(0, 0), pool_type='max', name='P6_subsampling')

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride64": p6, "stride32": p5_conv, "stride16": p4_conv, "stride8": p3_conv, "stride4": p2_conv})

    return conv_fpn_feat, [p6, p5_conv, p4_conv, p3_conv, p2_conv]


def get_resnet_fpn_mask_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    # # shared parameters for predictions
    rpn_conv_weight = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_conv_bbox_bias')

    rcnn_fc6_weight = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('rcnn_fc_bbox_weight')
    rcnn_fc_bbox_bias = mx.symbol.Variable('rcnn_fc_bbox_bias')

    kp_conv_1_weight = mx.symbol.Variable('kp_conv_1_weight')
    kp_conv_1_bias = mx.symbol.Variable('kp_conv_1_bias')
    kp_conv_2_weight = mx.symbol.Variable('kp_conv_2_weight')
    kp_conv_2_bias = mx.symbol.Variable('kp_conv_2_bias')
    kp_conv_3_weight = mx.symbol.Variable('kp_conv_3_weight')
    kp_conv_3_bias = mx.symbol.Variable('kp_conv_3_bias')
    kp_conv_4_weight = mx.symbol.Variable('kp_conv_4_weight')
    kp_conv_4_bias = mx.symbol.Variable('kp_conv_4_bias')
    kp_conv_5_weight = mx.symbol.Variable('kp_conv_5_weight')
    kp_conv_5_bias = mx.symbol.Variable('kp_conv_5_bias')
    kp_conv_6_weight = mx.symbol.Variable('kp_conv_6_weight')
    kp_conv_6_bias = mx.symbol.Variable('kp_conv_6_bias')
    kp_conv_7_weight = mx.symbol.Variable('kp_conv_7_weight')
    kp_conv_7_bias = mx.symbol.Variable('kp_conv_7_bias')
    kp_conv_8_weight = mx.symbol.Variable('kp_conv_8_weight')
    kp_conv_8_bias = mx.symbol.Variable('kp_conv_8_bias')
    kp_deconv_1_weight = mx.symbol.Variable('kp_deconv_1_weight')
    # kp_deconv_1_bias = mx.symbol.Variable('kp_deconv_1_bias')

    num_fg_rois = int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION)

    rpn_rois_list = []
    rpn_cls_list = []
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s' % stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                        act_type="relu",
                                        name="rpn_relu")
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # ROI Proposal
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='rpn_cls_prob_reshape')
        rpn_rois, rpn_scores = mx.sym.contrib.Proposal(cls_prob=rpn_cls_prob_reshape,
                                                       bbox_pred=rpn_bbox_pred,
                                                       im_info=im_info,
                                                       rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N,
                                                       rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
                                                       feature_stride=stride,
                                                       output_score=True,
                                                       scales=tuple(config.ANCHOR_SCALES),
                                                       ratios=tuple(config.ANCHOR_RATIOS),
                                                       rpn_min_size=config.TEST.RPN_MIN_SIZE[config.RPN_FEAT_STRIDE.index(stride)],
                                                       threshold=config.TEST.RPN_NMS_THRESH)
        #rpn_cls_prob_dict.update({'cls_prob_stride%s' % stride: rpn_cls_prob_reshape})
        #rpn_bbox_pred_dict.update({'bbox_pred_stride%s' % stride: rpn_bbox_pred})
        rpn_rois_list.append(mx.sym.Reshape(data=rpn_rois, shape=(-1, 5)))
        rpn_cls_list.append(mx.sym.Reshape(data=rpn_scores, shape=(-1, 1)))

    # fpn rois
    # fpn_roi_feats = list()
    rpn_rois_concat = mx.sym.concat(*rpn_rois_list, dim=0, name="rpn_rois")
    rpn_scores_concat = mx.sym.concat(*rpn_cls_list, dim=0, name="rpn_cls_score")
    args_dict = dict()
    args_dict.update({'roi_scores': rpn_scores_concat,'rois': rpn_rois_concat,'op_type':'rpn_proposal'})
    rois = mx.symbol.Custom(**args_dict)
    # FPN roi pooling
    args_dict = {}
    for s in config.RCNN_FEAT_STRIDE:
        args_dict.update({'feat_stride%s' % s: conv_fpn_feat['stride%s' % s]})
    args_dict.update({'rois': rois,
                      'name': 'fpn_roi_pool',
                      'op_type': 'fpn_roi_pool',
                      'rcnn_strides': config.RCNN_FEAT_STRIDE,
                      'pool_h': 7,
                      'pool_w': 7})
    roi_pool_fpn = mx.symbol.Custom(**args_dict)

    # classification with fc layers
    flatten = mx.symbol.Flatten(data=roi_pool_fpn, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")

    # classification
    rcnn_cls_score = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_cls_weight,
                                              bias=rcnn_fc_cls_bias, num_hidden=num_classes)
    rcnn_cls_prob = mx.symbol.SoftmaxActivation(name='rcnn_cls_prob', data=rcnn_cls_score)
    # bounding box regression
    rcnn_bbox_pred = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_bbox_weight,
                                              bias=rcnn_fc_bbox_bias, num_hidden=num_classes * 4)

    # reshape output
    rcnn_cls_prob = mx.symbol.Reshape(data=rcnn_cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
    rcnn_bbox_pred = mx.symbol.Reshape(data=rcnn_bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
                                       name='bbox_pred_reshape')

    # we can control #rois within this op
    kp_rois, pred_boxes, score = \
        mx.symbol.Custom(data=data, label=rcnn_cls_prob, rois=rois, bbox_deltas=rcnn_bbox_pred, im_info=im_info,
                         op_type='MaskROI', num_classes=num_classes, topk=config.TEST.RPN_POST_NMS_TOP_N,
                         name='kp_roi')
    kp_rois_reshape = mx.sym.reshape(kp_rois, (-1, 5))

    args_dict = {}
    for s in config.RCNN_FEAT_STRIDE:
        args_dict.update({'feat_stride%s' % s: conv_fpn_feat['stride%s' % s]})

    args_dict.update({'rois': kp_rois_reshape, 'name': 'fpn_maskroi_pool',
                      'op_type': 'fpn_roi_pool',
                      'rcnn_strides': config.RCNN_FEAT_STRIDE,
                      'pool_h': 14, 'pool_w': 14})
    kp_roi_pool = mx.symbol.Custom(**args_dict)

    kp_conv_1 = mx.sym.Convolution(data=kp_roi_pool, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_1",
                                   weight=kp_conv_1_weight, bias=kp_conv_1_bias)
    kp_relu_1 = mx.sym.Activation(data=kp_conv_1, act_type="relu", name="kp_relu_1")

    kp_conv_2 = mx.sym.Convolution(data=kp_relu_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_2",
                                   weight=kp_conv_2_weight, bias=kp_conv_2_bias)
    kp_relu_2 = mx.sym.Activation(data=kp_conv_2, act_type="relu", name="kp_relu_2")

    kp_conv_3 = mx.sym.Convolution(data=kp_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_3",
                                   weight=kp_conv_3_weight, bias=kp_conv_3_bias)
    kp_relu_3 = mx.sym.Activation(data=kp_conv_3, act_type="relu", name="kp_relu_3")

    kp_conv_4 = mx.sym.Convolution(data=kp_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_4",
                                   weight=kp_conv_4_weight, bias=kp_conv_4_bias)
    kp_relu_4 = mx.sym.Activation(data=kp_conv_4, act_type="relu", name="kp_relu_4")

    kp_conv_5 = mx.sym.Convolution(data=kp_relu_4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_5",
                                   weight=kp_conv_5_weight, bias=kp_conv_5_bias)
    kp_relu_5 = mx.sym.Activation(data=kp_conv_5, act_type="relu", name="kp_relu_5")

    kp_conv_6 = mx.sym.Convolution(data=kp_relu_5, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_6",
                                   weight=kp_conv_6_weight, bias=kp_conv_6_bias)
    kp_relu_6 = mx.sym.Activation(data=kp_conv_6, act_type="relu", name="kp_relu_6")

    kp_conv_7 = mx.sym.Convolution(data=kp_relu_6, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_7",
                                   weight=kp_conv_7_weight, bias=kp_conv_7_bias)
    kp_relu_7 = mx.sym.Activation(data=kp_conv_7, act_type="relu", name="kp_relu_7")

    kp_conv_8 = mx.sym.Convolution(data=kp_relu_7, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_8",
                                   weight=kp_conv_8_weight, bias=kp_conv_8_bias)
    kp_relu_8 = mx.sym.Activation(data=kp_conv_8, act_type="relu", name="kp_relu_8")

    kp_deconv_1 = mx.sym.Deconvolution(data=kp_relu_8, kernel=(4, 4), stride=(2, 2), num_filter=17, name="kp_deconv_1",
                                       target_shape=(28, 28), weight=kp_deconv_1_weight)
    kp_upsample = mx.sym.UpSampling(data=kp_deconv_1, scale=2, num_filter=17, sample_type='bilinear',
                                    name='kp_upsample')
    kp_reshape = kp_upsample.reshape((-1, config.KEYPOINT.MAPSIZE*config.KEYPOINT.MAPSIZE))
    kp_prob = mx.symbol.SoftmaxActivation(data=kp_reshape, name='kp_prob')
    kp_prob = kp_prob.reshape((config.TEST.BATCH_IMAGES, -1, 17, config.KEYPOINT.MAPSIZE*config.KEYPOINT.MAPSIZE), name='kp_prob_reshape')

    #group = mx.symbol.Group([rois, rcnn_bbox_pred, rcnn_cls_prob, mask_prob])
    group = mx.symbol.Group([pred_boxes, score, kp_prob])
    return group

def get_resnet_fpn_rpn(num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)

    # shared convolutional layers, top down
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    # shared parameters for predictions
    rpn_conv_weight = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_conv_bbox_bias')

    rpn_cls_score_list = []
    rpn_bbox_pred_list = []
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s' % stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # prepare rpn data
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1),
                                                  name="rpn_cls_score_reshape_stride%s" % stride)
        rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                  shape=(0, 0, -1),
                                                  name="rpn_bbox_pred_reshape_stride%s" % stride)

        rpn_bbox_pred_list.append(rpn_bbox_pred_reshape)
        rpn_cls_score_list.append(rpn_cls_score_reshape)

    # concat output of each level
    rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list, dim=2)
    rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list, dim=2)

    # loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_concat,
                                           label=rpn_label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rpn_cls_prob')

    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                           data=(rpn_bbox_pred_concat - rpn_bbox_target))

    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                    grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    rpn_group = [rpn_cls_prob, rpn_bbox_loss]
    group = mx.symbol.Group(rpn_group)
    return group


def get_resnet_fpn_rpn_test(num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    # # shared parameters for predictions
    rpn_conv_weight = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_conv_bbox_bias')

    rpn_rois_list = []
    rpn_scores_list = []
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s' % stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # ROI Proposal
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='rpn_cls_prob_reshape')

        rpn_rois, rpn_scores = mx.sym.contrib.Proposal(cls_prob=rpn_cls_prob_reshape,
                                                       bbox_pred=rpn_bbox_pred,
                                                       im_info=im_info,
                                                       rpn_pre_nms_top_n=config.TRAIN.PROPOSAL_PRE_NMS_TOP_N,
                                                       rpn_post_nms_top_n=config.TRAIN.PROPOSAL_POST_NMS_TOP_N,
                                                       feature_stride=stride,
                                                       output_score=True,
                                                       scales=tuple(config.ANCHOR_SCALES),
                                                       ratios=tuple(config.ANCHOR_RATIOS),
                                                       rpn_min_size=config.TEST.RPN_MIN_SIZE[config.RPN_FEAT_STRIDE.index(stride)],
                                                       threshold=config.TEST.RPN_NMS_THRESH)

        rpn_rois_list.append(rpn_rois)
        rpn_scores_list.append(rpn_scores)

    rpn_rois_concat = mx.sym.concat(*rpn_rois_list, dim=1, name="rpn_rois")
    rpn_scores_concat = mx.sym.concat(*rpn_scores_list, dim=1, name="rpn_scores")

    return mx.sym.Group([rpn_rois_concat, rpn_scores_concat])


def get_resnet_fpn_maskrcnn(num_classes=config.NUM_CLASSES):
    rcnn_feat_stride = config.RCNN_FEAT_STRIDE

    # create symbol for inputs and labels
    # then flatten the first two axis, MXNet does not allow input data to have different batch size
    data        = mx.sym.var("data")
    label       = mx.sym.var("label").reshape(shape=(-1, ))
    bbox_target = mx.sym.var("bbox_target").reshape(shape=(-1, 4 * num_classes))
    bbox_weight = mx.sym.var("bbox_weight").reshape(shape=(-1, 4 * num_classes))
    # mask_target = mx.sym.var("mask_target").reshape(shape=(-1, num_classes, 28, 28))
    keypoint_target = mx.sym.var("keypoint_target").reshape(shape=(-1,))
    rois = dict()
    for s in rcnn_feat_stride:
        rois["stride%s" % s] = mx.sym.var("rois_stride%s" % s).reshape(shape=(-1, 5))

    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)

    # shared convolutional layers, top down
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    # fpn rois
    fpn_roi_feats = list()
    for stride in rcnn_feat_stride:
        feat_lvl = conv_fpn_feat["stride%s" % stride]
        rois_lvl = rois["stride%s" % stride]
        if config.ROIALIGN:
            roi_feat = mx.sym.contrib.ROIAlign_v2(
                name="roi_pool", data=feat_lvl, rois=rois_lvl, pooled_size=(7, 7), spatial_scale=1.0 / stride)
        else:
            roi_feat = mx.sym.ROIPooling(
                name="roi_pool", data=feat_lvl, rois=rois_lvl, pooled_size=(7, 7), spatial_scale=1.0 / stride)
        fpn_roi_feats.append(roi_feat)
    roi_pool = mx.sym.add_n(*fpn_roi_feats)

    mask_roi_feats = list()
    for stride in rcnn_feat_stride:
        feat_lvl = conv_fpn_feat["stride%s" % stride]
        rois_lvl = rois["stride%s" % stride]
        mask_feat = mx.sym.contrib.ROIAlign_v2(
            name="roi_mask", data=feat_lvl, rois=rois_lvl, pooled_size=(14, 14), spatial_scale=1.0 / stride)
        mask_roi_feats.append(mask_feat)
    # merge rois from different levels
    # each mask_feat is 512 * 256 * 14 * 14
    # add them element wise
    mask_pool = mx.sym.add_n(*mask_roi_feats)

    ######################################################################
    # rcnn branch
    ######################################################################
    flatten = mx.sym.Flatten(data=roi_pool, name="flatten")
    fc6 = mx.sym.FullyConnected(data=flatten, num_hidden=1024, name='rcnn_fc6')
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.sym.FullyConnected(data=drop6, num_hidden=1024, name='rcnn_fc7')
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")

    cls_score = mx.sym.FullyConnected(data=relu7, num_hidden=num_classes, name="rcnn_fc_cls")
    cls_prob = mx.sym.SoftmaxOutput(data=cls_score, label=label, normalization="valid", multi_output=True,
                                    use_ignore=True, ignore_label=-1, name="rcnn_cls_prob")

    bbox_pred = mx.sym.FullyConnected(data=relu7, num_hidden=num_classes * 4, name="rcnn_fc_bbox")
    bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name="rcnn_bbox_loss", scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name="bbox_loss", data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    rcnn_group = [cls_prob, bbox_loss]

    # ###################################################################
    # # debug branch
    # ###################################################################
    # rois_add_n = mx.sym.add_n(*rois.values())
    # bug = mx.sym.Custom(op_type="Debug", num_args=4, data=data, data1=rois_add_n, data2=mask_target, data3=label,
    #                     pos="demo_rcnn_proposal")
    # bug = mx.sym.MakeLoss(bug)

    # ###################################################################
    # # mask branch
    # ###################################################################
    # num_fg_rois = int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION)

    # mask_pool = mask_pool.reshape(shape=(-1, config.TRAIN.BATCH_ROIS, 256, 14, 14))
    # mask_pool = mask_pool.slice_axis(axis=1, begin=0, end=num_fg_rois)
    # mask_pool = mask_pool.reshape(shape=(-1, 256, 14, 14))

    # mask_pool_instance = mask_pool.slice_axis(axis=0, begin=0, end=num_fg_rois)
    #
    # mask_conv_1 = mx.sym.Convolution(data=mask_pool_instance, kernel=(3, 3), pad=(1, 1), num_filter=256, name="mask_conv_1")
    # mask_relu_1 = mx.sym.Activation(data=mask_conv_1, act_type="relu", name="mask_relu_1")
    #
    # mask_conv_2 = mx.sym.Convolution(data=mask_relu_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="mask_conv_2")
    # mask_relu_2 = mx.sym.Activation(data=mask_conv_2, act_type="relu", name="mask_relu_2")
    #
    # mask_conv_3 = mx.sym.Convolution(data=mask_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="mask_conv_3")
    # mask_relu_3 = mx.sym.Activation(data=mask_conv_3, act_type="relu", name="mask_relu_3")
    #
    # mask_conv_4 = mx.sym.Convolution(data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="mask_conv_4")
    # mask_relu_4 = mx.sym.Activation(data=mask_conv_4, act_type="relu", name="mask_relu_4")
    #
    # mask_deconv_1 = mx.sym.Deconvolution(data=mask_relu_4, kernel=(2, 2), stride=(2, 2), num_filter=256,
    #                                      name="mask_deconv_1")
    # mask_relu_5 = mx.sym.Activation(data=mask_deconv_1, act_type="relu", name="mask_relu_5")
    #
    # mask_deconv_2 = mx.sym.Convolution(data=mask_relu_5, kernel=(1, 1), num_filter=num_classes, name="mask_deconv_2")
    #
    # mask_deconv_2 = mask_deconv_2.reshape((1, -1))
    # mask_target = mask_target.reshape((1, -1))
    # mask_loss = mx.sym.contrib.SigmoidCrossEntropy(data=mask_deconv_2, label=mask_target,
    #                                                 grad_scale=cfg.MASKRCNN.MASK_LOSS, name="mask_output")

    # mask_loss = mx.sym.MultiLogistic(data=mask_deconv_2, label=mask_target,
    #                                               grad_scale=cfg.MASKRCNN.MASK_LOSS, name="mask_output")

    # mask_group = [mask_loss]

    ####################################################################
    # keypoint branch
    ####################################################################
    # num_fg_rois = int(config.KEYPOINT.fg_num)
    num_fg_rois = int(config.TRAIN.BATCH_ROIS * config.TRAIN.FG_FRACTION)
    keypoint_pool = mask_pool.slice_axis(axis=0, begin=0, end=num_fg_rois)

    kp_conv_1 = mx.sym.Convolution(data=keypoint_pool, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_1")
    kp_relu_1 = mx.sym.Activation(data=kp_conv_1, act_type="relu", name="kp_relu_1")

    kp_conv_2 = mx.sym.Convolution(data=kp_relu_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_2")
    kp_relu_2 = mx.sym.Activation(data=kp_conv_2, act_type="relu", name="kp_relu_2")

    kp_conv_3 = mx.sym.Convolution(data=kp_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_3")
    kp_relu_3 = mx.sym.Activation(data=kp_conv_3, act_type="relu", name="kp_relu_3")

    kp_conv_4 = mx.sym.Convolution(data=kp_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_4")
    kp_relu_4 = mx.sym.Activation(data=kp_conv_4, act_type="relu", name="kp_relu_4")

    kp_conv_5 = mx.sym.Convolution(data=kp_relu_4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_5")
    kp_relu_5 = mx.sym.Activation(data=kp_conv_5, act_type="relu", name="kp_relu_5")

    kp_conv_6 = mx.sym.Convolution(data=kp_relu_5, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_6")
    kp_relu_6 = mx.sym.Activation(data=kp_conv_6, act_type="relu", name="kp_relu_6")

    kp_conv_7 = mx.sym.Convolution(data=kp_relu_6, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_7")
    kp_relu_7 = mx.sym.Activation(data=kp_conv_7, act_type="relu", name="kp_relu_7")

    kp_conv_8 = mx.sym.Convolution(data=kp_relu_7, kernel=(3, 3), pad=(1, 1), num_filter=512, name="kp_conv_8")
    kp_relu_8 = mx.sym.Activation(data=kp_conv_8, act_type="relu", name="kp_relu_8")

    kp_deconv_1 = mx.sym.Deconvolution(data=kp_relu_8, kernel=(4, 4), stride=(2, 2), num_filter=17, name="kp_deconv_1",
                                       target_shape=(28, 28))
    kp_upsample = mx.sym.UpSampling(data=kp_deconv_1, scale=2, num_filter=17, sample_type='bilinear', name='kp_upsample')

    kp_prob = kp_upsample.reshape((num_fg_rois*17, config.KEYPOINT.MAPSIZE*config.KEYPOINT.MAPSIZE))

    # kp_deconv_1 = mx.sym.Deconvolution(data=kp_relu_8, kernel=(2, 2), stride=(2, 2), num_filter=512,
    #                                      name="kp_deconv_1")
    # kp_relu_9 = mx.sym.Activation(data=kp_deconv_1, act_type="relu", name="kp_relu_9")
    #
    # kp_upsample = mx.sym.UpSampling(data=kp_relu_9, scale=2, num_filter=512, sample_type='bilinear', name='kp_upsample')
    #
    # kp_deconv_2 = mx.sym.Convolution(data=kp_upsample, kernel=(1, 1), num_filter=17, name="kp_deconv_2")
    #
    # kp_deconv_2 = kp_deconv_2.reshape((num_fg_rois*17, config.KEYPOINT.MAPSIZE*config.KEYPOINT.MAPSIZE))

    # keypoint_target = keypoint_target.reshape((num_fg_rois*17,))

    kp_loss = mx.sym.SoftmaxOutput(data=kp_prob, label=keypoint_target, multi_output=True, use_ignore=True,
                                   ignore_label=-1, name='kp_output', normalization='valid')
    kp_group = [kp_loss]

    return mx.sym.Group(rcnn_group + kp_group)


if __name__ == '__main__':
    rcnn.config.generate_config('resnet_fpn', 'coco')
    sym = get_resnet_fpn_maskrcnn(2)
    shape_dict = {
        'bbox_target': (1L, 512L, 8L),
        'bbox_weight': (1L, 512L, 8L),
        'data': (1L, 3L, 800L, 1333L),
        'label': (1L, 512L),
        'mask_target': (1L, 128L, 2L, 28L, 28L),
        'rois_stride16': (1L, 512L, 5L),
        'rois_stride32': (1L, 512L, 5L),
        'rois_stride4': (1L, 512L, 5L),
        'rois_stride8': (1L, 512L, 5L),
        'keypoint_target': (1L, 128L, 17L, 1L)
    }

    arg_shape, out_shape, aux_shape = sym.infer_shape(**shape_dict)
    import pprint
    pprint.pprint(out_shape)
