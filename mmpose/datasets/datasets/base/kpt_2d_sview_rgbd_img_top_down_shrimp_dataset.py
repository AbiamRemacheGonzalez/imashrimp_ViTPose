import copy
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from imashrimp_ViTPose.mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme, keypoint_pck_accuracy, _calc_distances, keypoint_coordinate_errors)
from imashrimp_ViTPose.mmpose.datasets import DatasetInfo
from imashrimp_ViTPose.mmpose.datasets.pipelines import Compose


class Kpt2dSviewRgbdImgTopDownShrimpDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_prefix_depth,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 coco_style=True,
                 test_mode=False):

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.img_prefix_depth = img_prefix_depth
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.ann_info['max_num_joints'] = data_cfg.get('max_num_joints', None)
        self.ann_info['dataset_idx'] = data_cfg.get('dataset_idx', 0)

        self.ann_info['use_different_joint_weights'] = data_cfg.get(
            'use_different_joint_weights', False)

        if dataset_info is None:
            raise ValueError(
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.')

        dataset_info = DatasetInfo(dataset_info)

        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id, self.id2depthname, self.depthname2id = self._get_mapping_id_name(self.coco.imgs)

        self.db = []

        self.pipeline = Compose(self.pipeline)

    @staticmethod
    def _get_mapping_id_name(imgs):
        id2name = {}
        name2id = {}
        id2depthname = {}
        depthname2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id
            depthname = file_name.replace('.png', '.npy')
            depthname = depthname.replace('C', 'D', 1)
            depthname2id[depthname] = image_id
            id2depthname[image_id] = depthname

        return id2name, name2id, id2depthname, depthname2id

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get the normalize factor. generally inter-ocular distance measured
        as the Euclidean distance between the outer corners of the eyes is
        used. This function should be overrode, to measure NME.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Returns:
            np.ndarray[N, 2]: normalized factor
        """
        return np.ones([gts.shape[0], 2], dtype=np.float32)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.01,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
            pck_thr (float): PCK threshold, default as 0.01.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []
        threshold_bbox_for_pcke = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))
            if 'PCKe' in metrics:
                threshold_bbox_for_pcke.append(np.array([1, 1]))
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        threshold_bbox_for_pcke = np.array(threshold_bbox_for_pcke)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('PCK', pck))

        if 'PCKe' in metrics:
            norm = _calc_distances(outputs, gts, masks, threshold_bbox)
            no_norm = _calc_distances(outputs, gts, masks, threshold_bbox_for_pcke)
            error_one = norm / no_norm
            mean_one = error_one.mean()
            pixels_thr = pck_thr / mean_one

            thrs = [1, 2, 3, 4, 5, 10, 15, 20]
            _, pcke, _ = keypoint_pck_accuracy(outputs, gts, masks, pixels_thr, threshold_bbox_for_pcke)
            info_str.append(('PCKe_' + str(pck_thr) + "_" + str(round(pixels_thr, 2)), pcke))
            for thr in thrs:
                _, pcke, _ = keypoint_pck_accuracy(outputs, gts, masks, round(thr), threshold_bbox_for_pcke)
                info_str.append(('PCKe_' + str(round(thr)), pcke))
            # distances = _calc_distances(outputs, gts, masks, threshold_bbox_for_pcke)
            # info_str.append(('PCKdis', distances))

        if 'PCKh' in metrics:
            _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks, pckh_thr,
                                               threshold_head_box)
            info_str.append(('PCKh', pckh))

        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
                                                 auc_nor)))

        if 'EPE' in metrics:
            info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        if 'NME' in metrics:
            normalize_factor = self._get_normalize_factor(
                gts=gts, box_sizes=box_sizes)
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        if 'CEpj' in metrics:
            results_list = []
            num_keypoints = gts.shape[1]
            for i in range(gts.shape[1]):
                y_pred = outputs[:, i, :]
                y_true = gts[:, i, :]
                point_name = i + 1 if num_keypoints > 22 else i + 2
                result = keypoint_coordinate_errors(y_pred, y_true, point_name)
                results_list.append(result)
                # info_str.append(('EPEpj_' + str(point_name), str(round(result['EPE'], 2)) + "±" + str(round(result['SD(EPE)'], 2)) + "px"))
                # info_str.append(('MAPEpj_' + str(point_name), str(round(result['MAPE'], 2)) + "%"))

            y_pred_total = outputs.reshape(-1, 2)
            y_true_total = gts.reshape(-1, 2)
            point_name = f"General {num_keypoints}KP"
            result = keypoint_coordinate_errors(y_pred_total, y_true_total, point_name)
            results_list.append(result)
            info_str.append(('EPE_' + str(point_name),  str(round(result['EPE'], 2)) + "±" + str(round(result['SD(EPE)'], 2)) + "px"))
            info_str.append(('MAPE_' + str(point_name), str(round(result['MAPE'], 2)) + "%"))

            if num_keypoints == 23:
                y_pred = outputs[:, 1:23, :]
                y_true = gts[:, 1:23, :]
                y_pred_total = y_pred.reshape(-1, 2)
                y_true_total = y_true.reshape(-1, 2)
                point_name = f"General 22KP"
                result = keypoint_coordinate_errors(y_pred_total, y_true_total, point_name)
                results_list.append(result)
                info_str.append(('EPE_' + str(point_name), str(round(result['EPE'], 2)) + "±" + str(round(result['SD(EPE)'], 2)) + "px"))
                info_str.append(('MAPE_' + str(point_name), str(round(result['MAPE'], 2)) + "%"))
            info_str.append(('CEpj', results_list))  # Just to save it

        return info_str

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts