"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np

from data.imagenet import get_image_ids
from data.imagenet import get_bounding_boxes
from data.imagenet import get_image_sizes
from data.imagenet import get_partitions
from data.imagenet import get_class_labels_and_pred
from project_utils.utils import check_scoremap_validity
from project_utils.utils import check_box_convention


_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
_PARTITION_MAP = {0:'known', 1:'nov_s', 2:'nov_d'}



def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list




class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, gcam_threshold_list,
                 iou_threshold_list, partitions, multi_contour_eval):
        self.metadata = metadata
        self.gcam_threshold_list = gcam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_partitions = partitions
        self.dataset_name = dataset_name
        self.multi_contour_eval = multi_contour_eval

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        # self.cnt = {'all':0, 'known':0, 'nov_s':0, 'nov_d':0}
        # self.cnt_clus = {'all':0, 'known':0, 'nov_s':0, 'nov_d':0}

        self.cnt = {partition:0 for partition in self.dataset_partitions}
        self.cnt_clus = {partition:0 for partition in self.dataset_partitions}

        self.num_correct = \
            {partition: {iou_threshold: np.zeros(len(self.gcam_threshold_list))
                        for iou_threshold in self.iou_threshold_list}
            for partition in self.dataset_partitions}

        self.num_correct_top1 = \
            {partition: {iou_threshold: np.zeros(len(self.gcam_threshold_list))
                        for iou_threshold in self.iou_threshold_list}
             for partition in self.dataset_partitions}

        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)
        self.target_and_pred = get_class_labels_and_pred(self.metadata)
        self.partition = get_partitions(self.metadata)

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, image_id):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        target, pred = self.target_and_pred[image_id]
        partition = self.partition[image_id]
        partition = _PARTITION_MAP[partition]


        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.gcam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        # (N_threshold) --> (num_all_boxes, 4) <==> (num_a, 4)
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        # num_a x num_b
        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        # find the max iou match with different gt boxes,
        # and then find the max iou boxes in a threshold.
        # nr_box: number of each threshold
        # sliced_multiple_iou

        idx = 0
        sliced_multiple_iou = []  # len: 1000
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box
        # record clus_acc
        if target == pred:
            self.cnt_clus['all'] += 1
            self.cnt_clus[partition] += 1


        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]
            # record loc_acc
            self.num_correct['all'][_THRESHOLD][correct_threshold_indices] += 1
            self.num_correct[partition][_THRESHOLD][correct_threshold_indices] += 1
            # record clus_loc_acc
            if target == pred:
                self.num_correct_top1['all'][_THRESHOLD][correct_threshold_indices] += 1
                self.num_correct_top1[partition][_THRESHOLD][correct_threshold_indices] += 1


        self.cnt['all'] += 1
        self.cnt[partition] += 1

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        loc_acc = {partition:[] for partition in self.dataset_partitions}
        clus_loc_acc = {partition:[] for partition in self.dataset_partitions}
        clus_acc = {}
        for partition in self.dataset_partitions:
            for _THRESHOLD in self.iou_threshold_list:
                localization_accuracies_all = self.num_correct['all'][_THRESHOLD] * 100. / float(self.cnt['all'])
                loc_acc_max_index = np.where(localization_accuracies_all==localization_accuracies_all.max())
                cluster_localization_accuracies_all = self.num_correct['all'][_THRESHOLD] * 100. / float(self.cnt['all'])
                clus_loc_acc_max_index = np.where(cluster_localization_accuracies_all==cluster_localization_accuracies_all.max())

                # using the best threshold
                localization_accuracies = self.num_correct[partition][_THRESHOLD] * 100. / float(self.cnt[partition])
                cluster_localization_accuracies = self.num_correct_top1[partition][_THRESHOLD] * 100. / float(self.cnt[partition])
                loc_acc[partition].append(localization_accuracies[loc_acc_max_index].max())
                clus_loc_acc[partition].append(cluster_localization_accuracies[clus_loc_acc_max_index].max())


            clus_acc[partition] = self.cnt_clus[partition] / self.cnt[partition]

        return loc_acc, clus_loc_acc, clus_acc