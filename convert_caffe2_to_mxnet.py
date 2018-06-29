import cPickle as pkl
import numpy as np
import mxnet as mx
import pprint


def load_model_from_caffe2(model_path):
    with open(model_path) as fin:
        arg_params = pkl.load(fin)['blobs']
    _remove_momentum(arg_params)
    _remove_fc(arg_params)
    aux_params = _add_moving_mean_var(arg_params)
    _rename_bn(arg_params)
    _rename_conv_fc(arg_params)
    _swap_bgr_rgb(arg_params, 'conv1_weight')
    arg_params = {k: mx.nd.array(v) for k, v in arg_params.items()}
    aux_params = {k: mx.nd.array(v) for k, v in aux_params.items()}

    for k, v in sorted(arg_params.items()):
        print("{:<50}{}".format(k, v.shape))
    for k, v in sorted(aux_params.items()):
        print("{:<50}{}".format(k, v.shape))

    return arg_params, aux_params


def _swap_bgr_rgb(arg_params, first_conv):
    arg_params[first_conv] = arg_params[first_conv][:, ::-1, :, :]


def _add_moving_mean_var(arg_params):
    aux_params = {}
    for k in arg_params.keys():
        if 'bn_s' in k:
            aux_params[k[:-1] + 'moving_var'] = np.ones_like(arg_params[k])
        if 'bn_b' in k:
            aux_params[k[:-1] + 'moving_mean'] = np.zeros_like(arg_params[k])
    return aux_params


def _rename_conv_fc(arg_params):
    for k in arg_params.keys():
        if 'bn' not in k and '_w' in k:
            arg_params[k[:-1] + 'weight'] = arg_params[k]
            del arg_params[k]
        elif 'bn' not in k and '_b' in k:
            arg_params[k[:-1] + 'bias'] = arg_params[k]
            del arg_params[k]

def _rename_bn(arg_params):
    for k in arg_params.keys():
        if 'bn_s' in k:
            arg_params[k[:-1] + 'gamma'] = arg_params[k]
            del arg_params[k]
        elif 'bn_b' in k:
            arg_params[k[:-1] + 'beta'] = arg_params[k]
            del arg_params[k]


def _remove_momentum(arg_params):
    for k in arg_params.keys():
        if k.endswith('_momentum'):
            del arg_params[k]


def _remove_fc(arg_params):
    for k in arg_params.keys():
        if 'fc1000' in k:
            del arg_params[k]