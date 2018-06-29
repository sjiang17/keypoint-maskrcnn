from __future__ import print_function

import os
import json
import cv2
import numpy as np

from six.moves import range, zip, cPickle
from multiprocessing.dummy import Pool
from imdb import IMDB
from ..core.segms import flip_segms
from ..processing.bbox_transform import bbox_overlaps

# coco api
from ..pycocotools.coco import COCO
from ..pycocotools.cocoeval import COCOeval
from ..pycocotools.mask import encode as coco_mask_encode


class coco(IMDB):
    def __init__(self, image_set, root_path, data_path, load_memory=False, use_mask=False, panoptic=False):
        """
        fill basic information to initialize imdb
        :param image_set: train2014, val2014, test2015
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/coco'
        """
        super(coco, self).__init__('COCO', image_set, root_path, data_path)
        self.root_path = root_path
        self.data_path = data_path
        self.load_memory = load_memory
        self.use_mask = use_mask
        self.panoptic = panoptic
        self.coco = COCO(self._get_ann_file())

        # deal with class names
        # there are two kinds of indexes
        # train index which ranges from 0 to num_classes - 1
        # and coco index which comes with the annotation file
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        print('categories:', self.classes)
        self.num_classes = len(self.classes)
        print('num of categories:', self.num_classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print ('%s num_images %d' % (self.name, self.num_images))

        # deal with data name
        view_map = {'minival2014': 'val2014', 'valminusminival2014': 'val2014'}
        self.data_name = view_map[image_set] if image_set in view_map else image_set

    def _get_ann_file(self):
        """ self.data_path / annotations / instances_train2014.json """
        prefix = 'instances' if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.data_path, 'annotations', 'person', prefix + '_' + self.image_set + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def image_path_from_index(self, index):
        """ example: images / train2014 / COCO_train2014_000000119993.jpg """
        filename = '%012d.jpg' % (index)
        image_path = os.path.join(self.data_path, 'images', self.data_name, filename)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def create_roidb_from_box_list(self, proposal_list, gt_roidb):
        """
        given ground truth, prepare roidb
        :param proposal_list: (#img, #proposals, xyxy)
        :param gt_roidb: (#img), each item is a roirec
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        assert len(proposal_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for i in range(self.num_images):
            roi_rec = dict()
            roi_rec['image'] = gt_roidb[i]['image']
            roi_rec['height'] = gt_roidb[i]['height']
            roi_rec['width'] = gt_roidb[i]['width']

            proposals = proposal_list[i]
            if proposals.shape[1] == 5:
                proposals = proposals[:, :4]
            num_proposals = proposals.shape[0]

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_instances = gt_roidb[i]['ins_id']
                # n proposals and k gt_boxes => n * k overlap
                gt_overlaps = bbox_overlaps(proposals.astype(np.float32, copy=False),
                                            gt_boxes.astype(np.float32, copy=False))
                max_overlaps = gt_overlaps.max(axis=1)
                max_overlap_box_ids = gt_overlaps.argmax(axis=1)
                max_instances = gt_instances[max_overlap_box_ids]
                max_classes = gt_classes[max_overlap_box_ids]
                max_classes[max_overlaps == 0] = 0

                roi_rec.update({'boxes': proposals,
                                'gt_classes': np.zeros((num_proposals,), dtype=np.int32),
                                'max_classes': max_classes,
                                'max_overlaps': max_overlaps,
                                'ins_id': max_instances,
                                'flipped': False})
            else:
                roi_rec.update({'boxes': proposals,
                                'gt_classes': np.zeros((num_proposals,), dtype=np.int32),
                                'max_classes': np.zeros((num_proposals,), dtype=np.int32),
                                'max_overlaps': np.zeros((num_proposals,), dtype=np.float32),
                                'ins_id': np.zeros((num_proposals,), dtype=np.int32),
                                'flipped': False})
            # background roi => background class
            zero_indexes = np.where(roi_rec['max_overlaps'] == 0)[0]
            assert all(roi_rec['max_classes'][zero_indexes] == 0)
            # foreground roi => foreground class
            nonzero_indexes = np.where(roi_rec['max_overlaps'] > 0)[0]
            assert all(roi_rec['max_classes'][nonzero_indexes] != 0)

            roidb.append(roi_rec)

        return roidb

    def gt_roidb(self):
        if self.use_mask:
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_with_mask.pkl')
        else:
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print ('%s gt roidb loaded from %s' % (self.name, cache_file))
            return roidb

        # gt_roidb = [self._load_coco_annotation(index) for index in self.image_set_index]
        pool = Pool()
        gt_roidb = pool.map(self._load_coco_annotation, self.image_set_index)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print ('%s wrote gt roidb to %s' % (self.name, cache_file))

        return gt_roidb

    def _load_coco_annotation(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        img_desciption = self.coco.loadImgs(index)[0]
        width = img_desciption['width']
        height = img_desciption['height']

        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=None)
        ann_objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_ann_objs = []
        for ann_obj in ann_objs:
            if ann_obj['iscrowd']:
                continue
            x, y, w, h = ann_obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + w)
            y2 = min(height - 1, y + h)
            if ann_obj['area'] > 0 and x2 > x1 and y2 > y1:
                ann_obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_ann_objs.append(ann_obj)
        ann_objs = valid_ann_objs
        num_anns = len(ann_objs)

        gt_boxes = np.zeros((num_anns, 4), dtype=np.float32)
        gt_classes = np.zeros((num_anns, ), dtype=np.int32)
        overlaps = np.zeros((num_anns, self.num_classes), dtype=np.float32)
        ins_id = np.arange(num_anns)
        ins_poly = [_['segmentation'] for _ in ann_objs]  # polys are list of list, since instance can be multi-part

        for i, ann_obj in enumerate(ann_objs):
            cls = self._coco_ind_to_class_ind[ann_obj['category_id']]
            gt_boxes[i, :] = ann_obj['clean_bbox']
            gt_classes[i] = cls
            if ann_obj['iscrowd']:
                overlaps[i, :] = -1.0
            else:
                overlaps[i, cls] = 1.0

        if self.load_memory:
            image = cv2.imread(self.image_path_from_index(index))
        else:
            image = self.image_path_from_index(index)
        roi_rec = {'image': image,
                   'height': height,
                   'width': width,
                   'boxes': gt_boxes,
                   'gt_classes': gt_classes,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        if self.use_mask:
            roi_rec.update({'ins_id': ins_id, 'ins_poly': ins_poly})

        return roi_rec

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print('append flipped images to roidb')
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roirec = roidb[i]
            boxes = roirec['boxes'].copy()
            if boxes.shape[0] != 0:
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = roirec['width'] - oldx2 - 1
                boxes[:, 2] = roirec['width'] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all(),\
                    'img_name %s, width %d\n' % (roirec['image'], roirec['width']) + \
                    np.array_str(roirec['boxes'], precision=3, suppress_small=True)
            entry = {'image': roirec['image'],
                     'height': roirec['height'],
                     'width': roirec['width'],
                     'boxes': boxes,
                     'gt_classes': roirec['gt_classes'],
                     'max_classes': roirec['max_classes'],
                     'max_overlaps': roirec['max_overlaps'],
                     'flipped': True}
            if self.use_mask:
                entry.update({'ins_id': roirec['ins_id'],
                              'ins_poly': flip_segms(roirec['ins_poly'], roirec['height'], roirec['width'])})
            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def reorder(self, reorder_inds):
        self.image_set_index = [self.image_set_index[ind] for ind in reorder_inds]

    def evaluate_mask(self, results_pack):
        all_boxes = results_pack['all_boxes']
        all_masks = results_pack['all_masks']
        all_iminfos = [result['im_info'] for result in results_pack['results_list']]
        res_folder = os.path.join(self.cache_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'maskrcnn_%s_results.json' % self.image_set)
        self._write_coco_results(all_boxes, all_masks, all_iminfos, res_file)
        if 'test' not in self.image_set:
            self._do_python_eval(res_file, res_folder)

    def evaluate_detections(self, results_pack):
        """
        leave as empty, as is done in evaluate_mask
        :param results_pack:
        :return:
        """
        return

    def _write_coco_results(self, detections, masks, im_infos, res_file):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print ('collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1))
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_results_one_category(detections[cls_ind], masks[cls_ind], im_infos, coco_cat_id))
        print ('writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_results_one_category(self, boxes, all_masks, im_infos, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_set_index):
            dets = boxes[im_ind].astype(np.float)
            masks = all_masks[im_ind]
            if len(dets) == 0:
                continue
            scores = dets[:, -1]
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'segmentation': self._encode_mask([x1[k], y1[k], x2[k], y2[k]], masks[k], index),
                       'score': scores[k]} for k in range(dets.shape[0])]
            results.extend(result)
        return results

    def _do_python_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt)
        coco_eval.params.useSegm = False
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_metrics(coco_eval)
        eval_file = os.path.join(res_folder, '%s_%s_results.pkl' % ("bbox", self.image_set))
        with open(eval_file, 'wb') as f:
            cPickle.dump(coco_eval, f, cPickle.HIGHEST_PROTOCOL)
        print ('eval results saved to %s' % eval_file)

        coco_eval = COCOeval(self.coco, coco_dt)
        coco_eval.params.useSegm = True
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_metrics(coco_eval)
        eval_file = os.path.join(res_folder, '%s_%s_results.pkl' % ("seg", self.image_set))
        with open(eval_file, 'wb') as f:
            cPickle.dump(coco_eval, f, cPickle.HIGHEST_PROTOCOL)
        print ('eval results saved to %s' % eval_file)

    def _print_detection_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print ('~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~' % (IoU_lo_thresh, IoU_hi_thresh))
        print ('%-15s %5.1f' % ('all', 100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print ('%-15s %5.1f' % (cls, 100 * ap))

        print ('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _encode_mask(self, bbox, mask, im_id):
        im_ann = self.coco.loadImgs(im_id)[0]
        width = im_ann['width']
        height = im_ann['height']

        mask_img = np.zeros((height, width), dtype=np.uint8, order='F')  # col major
        bbox = map(int, bbox)

        if bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0:
            mask = cv2.resize(mask, (bbox[2] - bbox[0], bbox[3] - bbox[1]), interpolation=cv2.INTER_LINEAR)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask_img[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask

        return coco_mask_encode(mask_img)
