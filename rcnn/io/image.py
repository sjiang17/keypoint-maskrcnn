import numpy as np
import numpy.random as npr
import cv2
import os
import random
from ..config import config
import mxnet as mx


def get_image(roidb, use_random_scale, random_scale_range, fixed_multi_scales, pixel_mean):
    """
    Support multiple images
    preprocess image and return processed roidb
    :param roidb: a list of roirec
    :param use_random_scale:
    :param random_scale_range:
    :param fixed_multi_scales:
    :param pixel_mean:
    :return: a list of img as np.ndarray
    a list of roirec['image', 'flipped', 'boxes', 'im_info']
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        im = roi_rec['image']
        if isinstance(im, str):
            if os.path.exists(im):
                im = cv2.imread(im)  # read image in BGR mode, since coco contains grey image
            else:
                raise ValueError("image filed in roirec is neither an array nor a valid path: %s" % im)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        new_rec = roi_rec.copy()
        if use_random_scale:
            im_scale = npr.uniform(random_scale_range[0], random_scale_range[1])
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        else:
            # randomly pick a fixed scale
            scale_idx = random.randrange(len(fixed_multi_scales))
            shorter_edge = fixed_multi_scales[scale_idx][0]
            longer_edge = fixed_multi_scales[scale_idx][1]
            im, im_scale = resize(im, shorter_edge, longer_edge)

        im_tensor = transform(im, pixel_mean)
        processed_ims.append(im_tensor)
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)

    return processed_ims, processed_roidb


def get_image_and_mask(roidb, use_random_scale, random_scale_range, fixed_multi_scales, pixel_mean):
    """
    Support multiple images
    preprocess image and return processed roidb
    :param roidb: a list of roirec
    :param use_random_scale:
    :param random_scale_range:
    :param fixed_multi_scales:
    :param pixel_mean:
    :return: a list of img as np.ndarray
    a list of roirec['image', 'flipped', 'boxes', 'im_info']
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    processed_masks = []
    for i in range(num_images):
        roi_rec = roidb[i]
        im = roi_rec['image']
        if isinstance(im, str):
            if os.path.exists(im):
                im = cv2.imread(im)  # read image in BGR mode, since coco contains grey image
            else:
                raise ValueError("image filed in roirec is neither an array nor a valid path: %s" % im)
        mask = roi_rec['sem_seg']
        if isinstance(mask, str):
            if os.path.exists(mask):
                mask = cv2.imread(mask, -1)
                assert len(mask.shape) == 2, "label for semantic segmentation only has 1 channel"
            else:
                raise ValueError("sem_seg filed in roirec is neither an array nor a valid path: %s" % im)

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]

        new_rec = roi_rec.copy()
        if use_random_scale:
            im_scale = npr.uniform(random_scale_range[0], random_scale_range[1])
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        else:
            # randomly pick a fixed scale
            scale_idx = random.randrange(len(fixed_multi_scales))
            shorter_edge = fixed_multi_scales[scale_idx][0]
            longer_edge = fixed_multi_scales[scale_idx][1]
            im, im_scale = resize(im, shorter_edge, longer_edge)

        im_tensor = transform(im, pixel_mean)
        processed_ims.append(im_tensor)
        processed_masks.append(mask[None, :])
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)

    return processed_ims, processed_roidb, processed_masks


def resize(im, shorter_edge, longer_edge, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param shorter_edge: one dimensional size (the short side)
    :param longer_edge: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    im_shorter_edge = np.min(im_shape[0:2])
    im_longer_edge = np.max(im_shape[0:2])
    im_scale = float(shorter_edge) / float(im_shorter_edge)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_longer_edge) > longer_edge:
        im_scale = float(longer_edge) / float(im_longer_edge)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    return im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_device(im, pixel_means, ctx):
    """
    transform into mxnet tensor
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = mx.nd.zeros((3, im.shape[0], im.shape[1]), ctx)
    shape_0 = im.shape[0]
    shape_1 = im.shape[1]

    im = im.transpose((2, 0, 1))
    im = mx.nd.array(im, ctx, dtype='uint8')
    im = im.astype('float32')

    for i in range(3):
        im_tensor[i] = im[2 - i] - pixel_means[2 - i]
    im_tensor = im_tensor.reshape((1, 3, shape_0, shape_1))
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0, shape=None, out=None):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if out:
        all_tensor = out
    else:
        if shape:
            dimensions = shape
        if pad == 0:
            all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
        elif pad == 1:
            all_tensor = np.ones(tuple(dimensions), dtype=dtype)
        else:
            all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    elif ndim == 5:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3],
            :tensor.shape[4]] = tensor
    else:
        raise Exception("Sorry, {} not supported.".format(tensor_list[0].shape))
    return all_tensor
