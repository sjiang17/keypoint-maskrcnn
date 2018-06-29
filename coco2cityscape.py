"""
This script convert coco mask label to cityscape's format.
"""
from rcnn.pycocotools.coco import COCO
import numpy as np
import os
import os.path as osp
import PIL.Image as Image

dataDir = 'data/coco/'
dataTypes = ['train2014', 'val2014']

for dataType in dataTypes:
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    insSegDir = 'data/coco/ins_seg/'
    if not osp.exists(insSegDir):
        os.mkdir(insSegDir)

    insSegDir = os.path.join(insSegDir, dataType)
    if not osp.exists(insSegDir):
        os.mkdir(insSegDir)

    coco = COCO(annFile)

    imgIds = coco.getImgIds()
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgIds=[imgId], iscrowd=None)
        img_anno = coco.loadAnns(annIds)
        img_height = coco.imgs[imgId]['height']
        img_width = coco.imgs[imgId]['width']
        filename = coco.imgs[imgId]['file_name']
        print("Converting {}".format(os.path.join(dataDir + 'images', dataType, filename)))
        mask = np.zeros((img_height, img_width), dtype=np.int32)

        ins_id = 0
        for ann in img_anno:
            x, y, w, h = ann['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width - 1, x + w)
            y2 = min(img_height - 1, y + h)
            if ann['area'] > 0 and x2 > x1 and y2 > y1:
                m = coco.annToMask(ann)
                cat_id = ann['category_id']
                mask[m > 0] = cat_id * 500 + ins_id
                ins_id += 1
        ins_seg_im = Image.fromarray(mask)
        ins_seg_im.save(os.path.join(insSegDir, filename.replace('jpg', 'png')))
