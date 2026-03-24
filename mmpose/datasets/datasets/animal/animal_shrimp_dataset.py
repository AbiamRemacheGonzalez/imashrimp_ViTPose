import math
import os
import os.path as osp
from tqdm import tqdm
import cv2
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from imashrimp_mmcv.mmcv import Config, deprecated_api_warning
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbdImgTopDownShrimpDataset


@DATASETS.register_module()
class AnimalShrimpDataset(Kpt2dSviewRgbdImgTopDownShrimpDataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_prefix_depth,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/atrw.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            img_prefix_depth,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['use_different_joint_weights'] = False
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4], padding=1.0)

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            depth_file = osp.join(self.img_prefix_depth, self.id2depthname[img_id]) if self.img_prefix_depth else None
            rec.append({
                'image_file': image_file,
                'depth_file': depth_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, save_annotated_images=False, metric='PCK', **kwargs):
        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, '_evaluation_results.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, '_evaluation_results.json')

        # -------- Quantitative Evaluation --------
        quantitative_folder = osp.join(res_folder, 'quantitative_evaluation') if res_folder is not None else None
        if quantitative_folder is not None and not osp.exists(quantitative_folder):
            os.makedirs(quantitative_folder)
        #   COCO Evaluation (Default evaluation)
        coco_name_value = self.coco_evaluation(results, res_folder=quantitative_folder, metric=metric, **kwargs)
        #   Another Metrics Evaluation (PCK, PCKe, EPE per joint, etc.)
        general_name_value = self.general_evaluation(results, res_folder=quantitative_folder, metric=metric, **kwargs)

        # -------- Qualitative Evaluation --------
        qualitative_folder = osp.join(res_folder, 'qualitative_evaluation') if res_folder is not None else None
        if qualitative_folder is not None and not osp.exists(qualitative_folder):
            os.makedirs(qualitative_folder)
        if save_annotated_images:
            visualize_name_value = self.visualize_qualitative_results(results, res_folder=qualitative_folder, **kwargs)

        # -------- Extra outputs --------
        #   Distance Calculation (Adding calculation)
        distance_name_value = self.distance_calculation(results, res_folder=quantitative_folder, **kwargs)

        # Join the results
        name_value = OrderedDict()
        name_value.update(coco_name_value)
        name_value.update(general_name_value)
        name_value.update(distance_name_value)
        name_value.update(visualize_name_value if save_annotated_images else {})

        self._write_keypoint_results(name_value, res_file)

        return name_value

    def visualize_qualitative_results(self, results, res_folder=None, **kwargs):
        skeleton = kwargs["skeleton_order"]
        pred_images_info = self._get_pred_images_info(results)
        _distances_dict = defaultdict(list)

        for img_id in tqdm(pred_images_info.keys(), desc="Generating qualitative results"):#pred_images_info.keys():
            preds_info = pred_images_info[img_id]
            gts_info = self._load_coco_keypoint_annotation_kernel(img_id)
            for inst in range(len(preds_info)):
                pred_info = preds_info[inst]
                gt_info = gts_info[inst]

                gt_keypoints = gt_info['joints_3d'][:, :2]
                gt_bbox = gt_info['bbox']  # In Top-Down networks is the same in predicted and ground truth.

                pred_keypoints = pred_info['keypoints'][:, :2]
                pred_bbox = self._keypoints_to_coco_bbox(pred_keypoints)

                image_file = gt_info['image_file']
                image_name = osp.basename(image_file)

                heatmaps = pred_info['heatmap']  # 23 x 64 x 48

                source_image = self._read_image(image_file)

                # --------- Predicted Pose ----------------------
                image_with_pred_pose = self._draw_pose_on_image(source_image, pred_keypoints, pred_bbox, skeleton, text="Predicted Pose", line_color=(0, 255, 0), kp_color=(255, 0, 0))
                zoomed_image_with_pred_pose = self._crop_with_padding(image_with_pred_pose, pred_bbox)
                self._save_image(image_with_pred_pose, osp.join(res_folder, 'predicted_poses', f'pred_pose_{image_name}'))
                self._save_image(zoomed_image_with_pred_pose, osp.join(res_folder, 'predicted_poses', f'pred_pose_zoomed_{image_name}'))

                # --------- Ground Truth Pose --------------------
                image_with_gt_pose = self._draw_pose_on_image(source_image, gt_keypoints, gt_bbox, skeleton, text="Ground Truth Pose")
                zoomed_image_with_gt_pose = self._crop_with_padding(image_with_gt_pose, gt_bbox)
                self._save_image(image_with_gt_pose, osp.join(res_folder, 'ground_truth_poses', f'gt_pose_{image_name}'))
                self._save_image(zoomed_image_with_gt_pose, osp.join(res_folder, 'ground_truth_poses', f'gt_pose_zoomed_{image_name}'))

                # --------- Distances between key points --------- (pending implementation)
                # --------- Heatmaps ---------
                heatmap_overlay = self._overlay_heatmaps(source_image, heatmaps, gt_info['center'], gt_info['scale'])
                zoomed_image_with_heatmap_overlay = self._crop_with_padding(heatmap_overlay, gt_bbox)
                self._save_image(heatmap_overlay, osp.join(res_folder, 'heatmap_overlays', f'heatmap_overlay_{image_name}'))
                self._save_image(zoomed_image_with_heatmap_overlay, osp.join(res_folder, 'heatmap_overlays', f'heatmap_overlay_zoomed_{image_name}'))
                # --------- Image Composition --------- (pending implementation)
                composition = self._compose_images_vertically([image_with_gt_pose, heatmap_overlay, image_with_pred_pose])
                self._save_image(composition, osp.join(res_folder, 'compositions', f'composition_{image_name}'))

        name_value = OrderedDict([('visualize_qualitative_results', res_folder)])

        return name_value

    @staticmethod
    def _compose_images_vertically(images, spacing=0):
        widths = [img.shape[1] for img in images]
        heights = [img.shape[0] for img in images]
        max_width = max(widths)
        total_height = sum(heights) + spacing * (len(images) - 1)

        composition = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        current_y = 0
        for img in images:
            h, w = img.shape[:2]
            composition[current_y:current_y+h, :w] = img
            current_y += h + spacing

        return composition

    @staticmethod
    def _overlay_heatmaps(image, heatmaps, center, scale, alpha=0.7, colormap=cv2.COLORMAP_JET):
        """
        Overlays a set of heatmaps onto the original image using the center and scale.

        Args:
            image (np.ndarray): Original image of shape (H, W, 3).
            heatmaps (np.ndarray): Array of heatmaps of shape (num_keypoints, H_hm, W_hm).
            center (np.ndarray or list): [x, y] coordinates of the bbox center.
            scale (np.ndarray or list): [w, h] scale of the bbox (normalized by 200.0).
            alpha (float): Maximum transparency of the overlay (0.0 to 1.0).
            colormap (int): OpenCV colormap to use.

        Returns:
            np.ndarray: Final image with the overlaid heatmaps.
        """
        img_h, img_w = image.shape[:2]
        num_kp, hm_h, hm_w = heatmaps.shape

        # 1. Aggregate heatmaps (Take the maximum activation per pixel across all keypoints)
        hm_combined = np.max(heatmaps, axis=0)

        # 2. Normalize the heatmap to [0, 255] to apply color
        hm_min, hm_max = hm_combined.min(), hm_combined.max()
        if hm_max - hm_min > 0:
            hm_norm = ((hm_combined - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros_like(hm_combined, dtype=np.uint8)

        # 3. Colorize the heatmap
        hm_color = cv2.applyColorMap(hm_norm, colormap)

        # 4. Calculate the affine transformation (Heatmap -> Original Image)
        # Based on your code, the real scale is obtained by multiplying by 200.0
        box_w = scale[0] * 200.0
        box_h = scale[1] * 200.0

        # Top-left corner in the original image
        tl_x = center[0] - (box_w / 2.0)
        tl_y = center[1] - (box_h / 2.0)

        # Scaling factors
        sx = box_w / hm_w
        sy = box_h / hm_h

        # 2x3 affine transformation matrix
        M = np.array([
            [sx, 0, tl_x],
            [0, sy, tl_y]
        ], dtype=np.float32)

        # 5. Project the heatmap and its mask to the target resolution
        # cv2.warpAffine automatically handles image boundaries without throwing out-of-bounds errors
        hm_warped = cv2.warpAffine(hm_color, M, (img_w, img_h))
        mask_warped = cv2.warpAffine(hm_norm, M, (img_w, img_h))

        # 6. Smooth Alpha Blending
        # Convert the mask to a range of 0.0 to 'alpha' for smooth edge blending
        alpha_mask = (mask_warped.astype(np.float32) / 255.0) * alpha
        alpha_mask_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

        # Apply blending: Image * (1 - transp) + Heatmap * transp
        img_result = image.astype(np.float32) * (1 - alpha_mask_3d) + hm_warped.astype(np.float32) * alpha_mask_3d

        # Clip values between 0-255 and cast back to uint8
        img_result = np.clip(img_result, 0, 255).astype(np.uint8)

        return img_result

    @staticmethod
    def _save_image(image, filepath, is_rgb=False):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        img_to_save = image.copy()
        if is_rgb:
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(filepath, img_to_save)

        if not success:
            print(f"Error saving: {filepath}")

    @staticmethod
    def _crop_with_padding(image, bbox, padding=100):
        x, y, w, h = bbox
        img_height, img_width = image.shape[:2]
        start_x = int(x - padding)
        start_y = int(y - padding)
        end_x = int(x + w + padding)
        end_y = int(y + h + padding)
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img_width, end_x)
        end_y = min(img_height, end_y)
        cropped_image = image[start_y:end_y, start_x:end_x]
        return cropped_image

    @staticmethod
    def _draw_pose_on_image(image, keypoints, bbox, skeleton, line_color=(0, 0, 255), line_width=2, kp_color=(0, 255, 255), text=None):
        img_drawn = image.copy()
        skeleton = skeleton[1:] if len(keypoints) < 23 else skeleton[2:]  # No visualization of total_length and length of the abdomen. More visual.

        # ------------- Draw Bounding Box -------------
        padding = 50  # Padding around the bounding box for better visualization
        x, y, w, h = bbox
        img_height, img_width = img_drawn.shape[:2]

        start_x = max(0, int(x - padding))
        start_y = max(0, int(y - padding))
        end_x = min(img_width, int(x + w + padding))
        end_y = min(img_height, int(y + h + padding))

        start_point = (start_x, start_y)
        end_point = (end_x, end_y)

        cv2.rectangle(img_drawn, start_point, end_point, line_color, line_width)

        # ------------- Draw Bounding Box Text -------------
        font = cv2.FONT_HERSHEY_DUPLEX
        text_font_scale = 1.5
        text_thickness = 2
        if text is not None:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, text_font_scale, text_thickness)
            text_x = int(x)
            text_y = max(int(y) - 10, text_height + 5)
            bg_start_point = (text_x, text_y - text_height - 2)
            bg_end_point = (text_x + text_width, text_y + baseline + 2)
            cv2.rectangle(img_drawn, bg_start_point, bg_end_point, line_color, thickness=-1)
            cv2.putText(img_drawn, text, (text_x, text_y), font, fontScale=text_font_scale, color=(255, 255, 255),
                        thickness=text_thickness)

        # ------------- Draw Skeleton Lines -------------
        for connection in skeleton:
            kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1

            pt1 = (int(keypoints[kp1_idx, 0]), int(keypoints[kp1_idx, 1]))
            pt2 = (int(keypoints[kp2_idx, 0]), int(keypoints[kp2_idx, 1]))

            cv2.line(img_drawn, pt1, pt2, line_color, 4)

        # ------------- Draw Key points -------------
        for kp in keypoints:
            center = (int(kp[0]), int(kp[1]))
            cv2.circle(img_drawn, center, radius=10, color=(0, 0, 0), thickness=-1)
            cv2.circle(img_drawn, center, radius=8, color=kp_color, thickness=-1)

        return img_drawn

    @staticmethod
    def _show_image_plt(image, title="Visor de Imagen", is_bgr=True):
        """
        Muestra una imagen usando Matplotlib.
        Ideal para entornos interactivos como Jupyter Notebook o Google Colab.

        Args:
            image (np.ndarray): Imagen a mostrar.
            title (str): Título superior de la imagen.
            is_bgr (bool): Indica si la imagen está en formato BGR (por defecto en OpenCV).
        """
        import matplotlib.pyplot as plt
        # Matplotlib espera que los colores estén en orden RGB.
        # Si la imagen viene de OpenCV, la convertimos primero.
        if is_bgr:
            img_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_to_show = image.copy()

        plt.figure(figsize=(10, 10))  # Ajusta el tamaño de visualización
        plt.imshow(img_to_show)
        plt.title(title)
        plt.axis('off')  # Ocultamos los ejes con los números de los píxeles
        plt.show()

    @staticmethod
    def _read_image(path):
        image_np = cv2.imread(path)
        if image_np is None:
            # Probar a leer si tiene una dirección dentro.
            with open(path, "r", encoding="utf-8") as f:
                real_path = f.readline()
            image_np = cv2.imread(real_path)
        return image_np

    @staticmethod
    def _keypoints_to_coco_bbox(keypoints):
        points_x = keypoints[:, 0]
        points_y = keypoints[:, 1]

        x_min, x_max = np.min(points_x), np.max(points_x)
        y_min, y_max = np.min(points_y), np.max(points_y)

        width = x_max - x_min
        height = y_max - y_min

        return [float(x_min), float(y_min), float(width), float(height)]

    def distance_calculation(self, results, res_folder=None, **kwargs):
        skeleton = kwargs["skeleton_order"]
        skeleton_names = kwargs["skeleton_name"]

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_pixel_distance.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_pixel_distance.json')

        pred_images_info = self._get_pred_images_info(results)
        _distances_dict = defaultdict(list)

        for img_id in pred_images_info.keys():
            preds_info = pred_images_info[img_id]
            gts_info = self._load_coco_keypoint_annotation_kernel(img_id)
            for inst in range(len(preds_info)):
                pred_info = preds_info[inst]
                gt_info = gts_info[inst]

                gt_keypoints = gt_info['joints_3d'][:, :2]
                pred_keypoints = pred_info['keypoints'][:, :2]
                gt_bbox = gt_info['bbox']  # In Top-Down networks is the same in predicted and ground truth.
                image_file = gt_info['image_file']

                #  Ground Truth Distances
                gt_distances, gt_availability = self.get_keypoint_distances(gt_keypoints, skeleton)

                # Predicted Distances
                pred_distances, _ = self.get_keypoint_distances(pred_keypoints, skeleton)

                # Store the distances in the dictionary
                image_name = osp.basename(image_file)
                _distances_dict[image_name].append({
                    'image_file': image_file,
                    'image_id': img_id,
                    'gt_distances': gt_distances,
                    'pred_distances': pred_distances,
                    'bbox': gt_bbox,
                    'availability': gt_availability,
                    'skeleton_names': skeleton_names,
                    'skeleton_unions': skeleton,
                    'gt_keypoints': gt_keypoints.tolist(),
                    'pred_keypoints': pred_keypoints.tolist()
                })
        self._write_keypoint_results(_distances_dict, res_file)
        info_str = self._report_distance_metric(res_file)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    def _get_pred_images_info(self, rslts):
        _info = defaultdict(list)
        for result in rslts:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            heatmaps = result['output_heatmap']
            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                image_file = osp.join(self.img_prefix, self.id2name[image_id])
                _info[image_id].append({
                    'keypoints': preds[i],
                    'image_file': image_file,
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'heatmap': heatmaps[i]
                })
        return _info

    def _report_distance_metric(self, res_file):
        info_str = []

        with open(res_file, 'r') as fin:
            distances = json.load(fin)
        assert len(distances) == len(self.db)
        res_file_abs = res_file.split("2_pose_estimation/")[1]
        info_str.append(('pixel_distances_path', res_file_abs))
        return info_str

    def get_keypoint_distances(self, keypoints, skeleton):
        pixel_distances = []
        availability = []
        for union in skeleton:
            punto1 = [int(valor) for valor in keypoints[union[0] - 1]]  # int(valor)
            punto2 = [int(valor) for valor in keypoints[union[1] - 1]]
            if punto1 == [0, 0] or punto2 == [0, 0]:
                availability.append(0)
            else:
                availability.append(1)
            dis = self.euclidean_distance(punto1, punto2)
            final_dis = dis if dis > 0 else 0.0001
            pixel_distances.append(final_dis)
        return pixel_distances, availability

    @staticmethod
    def euclidean_distance(point_a, point_b):
        return math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def coco_evaluation(self, results, res_folder=None, metric='PCK', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP', 'PCK', 'PCKe', 'AUC', 'EPE', 'CEpj']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_coco_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_coco_keypoints.json')

        kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes_dict(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def general_evaluation(self, results, res_folder=None, metric='PCK', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP', 'PCK', 'PCKe', 'AUC', 'EPE', 'CEpj']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_general_keypoints.json')
            coordinate_error_file = osp.join(res_folder, '_evaluation_coordinate_error_per_joint.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_general_keypoints.json')
            coordinate_error_file = osp.join(tmp_folder.name, '_evaluation_coordinate_error_per_joint.json')

        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        keypoints_coordinate_error = name_value.get('CEpj', None)
        self._write_keypoint_results(keypoints_coordinate_error, coordinate_error_file)
        del name_value['CEpj']

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
            if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)', 'AP .95'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))
        info_str.append(("coco_keypoints_path", res_file))

        return info_str

    @staticmethod
    def _sort_and_unique_bboxes_dict(kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
