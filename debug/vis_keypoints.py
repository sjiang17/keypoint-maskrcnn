import cPickle
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import PIL.Image as Image
import os


def decode_kp(kp_label, bbox):
    if kp_label == -1:
        return 0, 0
    x = int(kp_label) % 56
    y = int(kp_label) / 56

    x = bbox[0] + float(x) / 56.0 * (bbox[2] - bbox[0])
    y = bbox[1] + float(y) / 56.0 * (bbox[3] - bbox[1])
    return x, y

folder = '/mnt/truenas/scratch/siyu/keypoint_maskrcnn/debug/save3'
save_folder = '/mnt/truenas/scratch/siyu/keypoint_maskrcnn/debug/keypoint_vis3'
data_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('data')])
label_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('label')])

colors = np.random.random(size=(17,3))

for j, (data_file, label_file) in enumerate(zip(data_files, label_files)):

    data = cPickle.load(open(data_file, 'r'))
    label = cPickle.load(open(label_file, 'r'))
    im = data[0].asnumpy()[0]
    im = im.transpose((1, 2, 0))
    im = im.astype(np.uint8)

    roi1 = data[1].asnumpy()[0][:, 1:]
    roi2 = data[2].asnumpy()[0][:, 1:]
    roi3 = data[3].asnumpy()[0][:, 1:]
    roi4 = data[4].asnumpy()[0][:, 1:]
    roi = roi1 + roi2 + roi3 + roi4

    keypoints = label[3].asnumpy()[0]

    plt.imshow(im)
    # im[:,::-1,:]
    # ind = 26
    for i in range(keypoints.shape[0]):
        bbox = roi[i]
        #     print bbox
        keypoint = keypoints[i]
        #     print keypoint
        xs = np.zeros(17)
        ys = np.zeros(17)
        for k, kp in enumerate(keypoint):
            xs[k], ys[k] = decode_kp(kp, bbox)
        #     print xs, ys
            plt.plot(xs[k], ys[k], 'o', color=colors[k])

    save_path = os.path.join(save_folder, '{}.jpg'.format(j))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=199)
    plt.close('all')
