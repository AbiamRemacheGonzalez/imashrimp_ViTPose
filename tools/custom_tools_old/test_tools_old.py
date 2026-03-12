from pylab import *
import json
import os
import pandas as pd
import math
import cv2
from xtcocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import seaborn as sns
from imashrimp_ViTPose.mmpose.core.evaluation.top_down_eval import (_get_max_preds)
from imashrimp_ViTPose.mmpose.core.evaluation.top_down_eval import (_calc_distances)

from pixel_to_cm_conversor.conversor import PixelToCentimeterConversor
from comparer.comparer import PopulationComparerByView
from searcher.utils import base as bs


def save_error_per_point_histogram_old(param, path_to_save):
    fig, axs = plt.subplots(param.shape[0], 1, figsize=(6, 60))
    min_error = np.min(param)
    max_error = np.max(param)
    for i in range(param.shape[0]):
        axs[i].hist(param[i, :], bins=param.shape[1], color='skyblue', edgecolor='black')
        axs[i].set_xlabel('error_distances')
        axs[i].set_ylabel('Frecuency')
        axs[i].set_title(f'Punto {i + 1}')
        axs[i].set_xlim(min_error, max_error)
        axs[i].set_ylim(0, param.shape[1] / 8)
    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.clf()


def save_error_per_point_histogram(param, path_to_save, metric="cm"):
    fig, axs = plt.subplots(param.shape[0], 1, figsize=(6, 60))
    min_error = np.min(param)
    max_error = np.max(param)

    for i in range(param.shape[0]):
        # Calculamos los estadísticos
        median = np.median(param[i, :])
        mean = np.mean(param[i, :])
        std_dev = np.std(param[i, :])

        # Histograma y KDE
        sns.histplot(param[i, :], bins=param.shape[1], color='skyblue', edgecolor='black', kde=True, ax=axs[i])

        # Añadir líneas para la mediana, media y desviación estándar
        axs[i].axvline(median, color='red', linestyle='--', label=f'Median: {median:.2f} {metric}')
        axs[i].axvline(mean, color='green', linestyle='-', label=f'Mean: {mean:.2f} {metric}')
        axs[i].axvline(mean - std_dev, color='orange', linestyle='-.',
                       label=f'SD (-1σ): {mean - std_dev:.2f} {metric}')
        axs[i].axvline(mean + std_dev, color='orange', linestyle='-.',
                       label=f'SD (+1σ): {mean + std_dev:.2f} {metric}')

        # Configuraciones de la gráfica
        axs[i].set_xlabel(f'Error Distances {metric}')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Punto {i + 1}')
        axs[i].set_xlim(min_error, max_error)
        axs[i].set_ylim(0, param.shape[1] / 8)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.clf()


def get_measure_info(real_measure_path, outputs):
    with open(real_measure_path) as archivo:
        source_info = json.load(archivo)
    day_info = source_info['day_info']
    for batch_item in outputs:
        image_paths = batch_item['image_paths']
        image_real_measure = {}
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_code = image_name.split('_')[3][:-4]
            image_day = day_info[image_code]
            img_dict = find_dict_by_param(source_info['images'], 'file_name', image_day)
            rm_id = img_dict['id']
            ann_dict = find_dict_by_int_param(source_info['annotations'], 'id', rm_id)
            point_a = (ann_dict['keypoints'][0], ann_dict['keypoints'][1])
            point_b = (ann_dict['keypoints'][3], ann_dict['keypoints'][4])
            pixels_per_cm = math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
            image_real_measure[image_name] = pixels_per_cm
        batch_item["image_real_measures"] = image_real_measure


def find_dict_by_param(lis, str_key, param):
    for dic in lis:
        if param in dic.get(str_key):
            return dic
    return None


def find_dict_by_int_param(lis, int_key, param):
    for dic in lis:
        if param == dic.get(int_key):
            return dic
    return None


def get_general_config(skeleton, skeleton_names, mode=0):
    config = {
        'bbox': "",
        'bbox_name': "",
        'bbox_color': (0, 0, 255),  # ROJO
        'bbox_line_width': 2,
        'bbox_text_font_scale': 1.5,
        'bbox_text_font_thickness': 2,
        'joint_line_width': 4,
        'joint_line_color': (0, 255, 255),  # AMARILLO
        'joint_text_color': (0, 0, 0),  # NEGRO
        'joint_text_font': cv2.FONT_HERSHEY_DUPLEX,
        'joint_text_font_scale': 0.8,
        'joint_text_font_thickness': 2,
        'circle_size': 10,
        'circle_color': (0, 0, 255),  # ROJO
        'keypoints': "",
        'cm_info': "",
        'skeleton': skeleton,
        'skeleton_names': skeleton_names,
    }
    if mode == 0:  # GROUND TRUTH
        return config
    elif mode == 1:  # PREDICTION
        config['bbox_color'] = (255, 0, 0)  # AZUL
        config['joint_line_color'] = (0, 255, 0)  # VERDE
        config['circle_color'] = (255, 0, 0)  # AZUL
        return config


def create_test_qualitative_images(
        outputs,
        dataset,
        checkpoint_name,
        cfg,
        args_radius=3,
        args_line_width=2,
        only_preds=False):
    args_out_img_root = cfg.work_dir + "/_test_qualitative_" + checkpoint_name
    args_img_root = dataset.img_prefix
    args_json_file = dataset.ann_file
    assert (args_out_img_root != '')
    os.makedirs(args_out_img_root, exist_ok=True)
    os.makedirs(os.path.join(args_out_img_root, "predictions_complete"), exist_ok=True)
    os.makedirs(os.path.join(args_out_img_root, "heatmap"), exist_ok=True)
    if not only_preds:
        os.makedirs(os.path.join(args_out_img_root, "ground_complete"), exist_ok=True)
        os.makedirs(os.path.join(args_out_img_root, "error"), exist_ok=True)
        os.makedirs(os.path.join(args_out_img_root, "combined"), exist_ok=True)

    ros_info = cfg.rostrum_info
    in_info = pd.read_csv(ros_info)
    in_info['name'] = in_info['ADDRESS'].apply(os.path.basename)
    ros_dict = dict(zip(in_info['name'], in_info['LABEL']))
    view_info = cfg.view_info
    in_info = pd.read_csv(view_info)
    in_info['name'] = in_info['ADDRESS'].apply(os.path.basename)
    view_dict = dict(zip(in_info['name'], in_info['LABEL']))

    predicted_measures = os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv')
    pm_df = pd.read_csv(predicted_measures)
    pm_df['super_code'] = pm_df['code'] + "_" + pm_df['point_of_view'].astype(str) + "_" + pm_df['angle'].astype(str)
    real_measures = os.path.join(args_out_img_root, f'rm_in_cm.csv')
    rm_df = pd.read_csv(real_measures)

    coco = COCO(args_json_file)
    img_keys = list(coco.imgs.keys())
    gt_imgs_info = [coco.loadImgs(image_id)[0] for image_id in img_keys]
    # skeleton = cfg.skeleton_order
    # skeleton_name = cfg.skeleton_name

    gt_config = get_general_config(cfg.skeleton_order, cfg.skeleton_name)
    pd_config = get_general_config(cfg.skeleton_order, cfg.skeleton_name, mode=1)
    predictions_in_cm = pd.read_csv(os.path.join(args_out_img_root, "predictions_converted_in_cm.csv"))
    with tqdm(total=len(coco.anns)) as pbar:
        for batch_item in outputs:
            for idx in range(len(batch_item['image_paths'])):
                image_name = os.path.basename(batch_item['image_paths'][idx])
                code = image_name.split('_')[3][:-4]
                super_code = image_name.split('_')[3][:-4] + "_" + image_name.split('_')[1] + "_" + \
                             image_name.split('_')[2]
                image_id = next((d for d in gt_imgs_info if d["file_name"] == image_name), None)['id']

                # prediction_info = pm_df.loc[pm_df['super_code'] == super_code]
                # real_info = rm_df.loc[rm_df['code'] == code]
                gt_config['cm_info'] = rm_df.loc[rm_df['code'] == code]
                pd_config['cm_info'] = pm_df.loc[pm_df['super_code'] == super_code]

                # gt_keypoint = get_gt_keypoints(coco, image_id)
                # gt_bbox = coco.anns[image_id]['bbox']
                # bbox_name = get_bbox_name(image_name, ros_dict, view_dict)

                gt_config['keypoints'] = get_gt_keypoints(coco, image_id)
                gt_config['bbox'] = coco.anns[image_id]['bbox']
                gt_config['bbox_name'] = get_bbox_name(image_name, ros_dict, view_dict)

                # pd_keypoint = batch_item['preds'][idx, :, :2]
                pd_heatmaps = batch_item['output_heatmap'][idx, :, :, :]
                pd_heatmaps = pd_heatmaps.reshape(1, pd_heatmaps.shape[0], pd_heatmaps.shape[1], pd_heatmaps.shape[2])

                pd_config['keypoints'] = batch_item['preds'][idx, :, :2]
                pd_config['bbox'] = get_bounding_box(pd_config['keypoints'])
                pd_config['bbox_name'] = get_bbox_name(image_name, ros_dict, view_dict)

                image_np = cv2.imread(os.path.join(args_img_root, image_name))
                if image_np is None:
                    # Probar a leer si tiene una dirección dentro.
                    with open(os.path.join(args_img_root, image_name), "r", encoding="utf-8") as f:
                        real_path = f.readline()
                    image_np = cv2.imread(real_path)

                out_file = os.path.join(args_out_img_root, 'combined', f'{image_name[:-4]}_test_comb.jpg')
                out_file_pred_complete = os.path.join(os.path.join(args_out_img_root, "predictions_complete"),
                                                      f'{image_name[:-4]}_test_complete.jpg')
                out_file_gt_complete = os.path.join(os.path.join(args_out_img_root, "ground_complete"),
                                                    f'{image_name[:-4]}_test_complete.jpg')
                out_file_err = os.path.join(os.path.join(args_out_img_root, "error"), f'{image_name[:-4]}_error.jpg')
                out_file_hm = os.path.join(os.path.join(args_out_img_root, "heatmap"), f'{image_name[:-4]}_ht.jpg')

                # --Zoom--
                zoom = get_image_zoom(pd_config['keypoints'], gt_config['keypoints'])
                # --Ground Truth--

                gt_image = get_image_with_key_points(image_np.copy(), gt_config, zoom=zoom)
                gt_image_complete = get_image_with_key_points(image_np.copy(), gt_config)
                # gt_image = get_gt_image(image_np.copy(), gt_bbox, gt_keypoint, skeleton, real_info, skeleton_name,
                #                         bbox_name, zoom=zoom, args_radius=args_radius, args_line_width=args_line_width)
                # gt_image_complete = get_gt_image(image_np.copy(), gt_bbox, gt_keypoint, skeleton, real_info,
                #                                  skeleton_name, bbox_name, zoom=None, args_radius=args_radius,
                #                                  args_line_width=args_line_width)
                if not only_preds:
                    cv2.imwrite(out_file_gt_complete, gt_image_complete)

                # --Prediction--
                pd_image = get_image_with_key_points(image_np.copy(), pd_config, zoom=zoom)
                pd_image_complete = get_image_with_key_points(image_np.copy(), pd_config)
                # pd_bbox = get_bounding_box(pd_keypoint)
                # pd_image = get_pd_image(image_np.copy(), pd_bbox, pd_keypoint, skeleton, prediction_info, skeleton_name,
                #                         bbox_name, zoom=zoom, args_radius=args_radius,
                #                         args_line_width=args_line_width)
                # pd_image_complete = get_pd_image(image_np.copy(), pd_bbox, pd_keypoint, skeleton, prediction_info,
                #                                  skeleton_name, bbox_name, zoom=None,
                #                                  args_radius=args_radius, args_line_width=args_line_width)
                cv2.imwrite(out_file_pred_complete, pd_image_complete)

                # --Error Image--
                dis_err_image = get_dis_err_image(image_np.copy(), gt_config['keypoints'], pd_config['keypoints'], zoom,
                                                  args_radius=args_radius, args_line_width=args_line_width)
                dis_err_image_nz = get_dis_err_image(image_np.copy(), gt_config['keypoints'], pd_config['keypoints'],
                                                     args_radius=args_radius, args_line_width=args_line_width)
                if not only_preds:
                    cv2.imwrite(out_file_err, dis_err_image_nz)

                # --Heatmap Image--
                ht_image = image_np.copy()
                if not only_preds:
                    image_hmj = get_image_with_all_joint_heatmaps(pd_heatmaps, pd_config['keypoints'], ht_image)
                    ht_image = get_superposition_images([image_hmj], ht_image)
                    cv2.imwrite(out_file_hm, ht_image)
                    ht_image = ht_image[zoom['y1']:zoom['y2'], zoom['x1']:zoom['x2'], :]
                    put_text_in_image(ht_image, "Heatmap Prediction")

                # --Saving the composition--
                if not only_preds:
                    cmp_image = np.concatenate((gt_image, dis_err_image, pd_image, ht_image), axis=0)
                else:
                    cmp_image = np.concatenate((gt_image, dis_err_image, pd_image), axis=0)
                if not only_preds:
                    cv2.imwrite(out_file, cmp_image)
                pbar.update(1)


def get_bbox_name(image_name, ros_dict, view_dict):
    if image_name not in ros_dict.values():
        ros_text = "Rostrum good"
    else:
        ros_text = "Rostrum break" if ros_dict[image_name] == 0 else "Rostrum good"
    if image_name not in view_dict.values():
        view_text = "Dorsal view"
    else:
        view_text = "Lateral view" if view_dict[image_name] == 0 else "Dorsal view"

    return f"{view_text}, {ros_text}"


def create_test_quantitative_results(outputs, dataset, checkpoint_name, cfg, only_preds=False, external_test=False):
    args_json_file = dataset.ann_file
    args_out_img_root = cfg.work_dir + "/_test_qualitative_" + checkpoint_name
    assert (args_out_img_root != '')
    os.makedirs(args_out_img_root, exist_ok=True)

    coco = COCO(args_json_file)
    img_keys = list(coco.imgs.keys())
    gt_imgs_info = [coco.loadImgs(image_id)[0] for image_id in img_keys]
    skeleton = cfg.skeleton_order

    num_kp = coco.anns[img_keys[0]]['num_keypoints']
    raw_info_gt = np.zeros((len(img_keys), num_kp, 2))
    raw_info_pd = np.zeros((len(img_keys), num_kp, 2))
    gt_dis_pix = np.zeros((len(img_keys), len(skeleton)))
    pd_dis_pix = np.zeros((len(img_keys), len(skeleton)))
    dis_availability = np.ones((len(img_keys), len(skeleton)))
    img_codes = []
    img_apa = []
    img_apv = []

    img_idx = 0
    for batch_item in outputs:
        for idx in range(len(batch_item['image_paths'])):
            image_name = os.path.basename(batch_item['image_paths'][idx])
            _, apv, apa, cod = image_name.split("_")

            img_apa.append(apa)
            img_apv.append(apv)
            img_codes.append(cod[:-4])

            image_id = next((d for d in gt_imgs_info if d["file_name"] == image_name), None)['id']
            gt_keypoint = get_gt_keypoints(coco, image_id)
            pd_keypoint = batch_item['preds'][idx, :, :2]

            # --Ground Truth--
            cgt_dis_pix, cgt_dis_av = bs.get_keypoint_distances(gt_keypoint, skeleton)
            gt_dis_pix[img_idx, :] = cgt_dis_pix
            dis_availability[img_idx, :] = cgt_dis_av
            raw_info_gt[img_idx, :, :] = gt_keypoint

            # --Prediction--
            cpd_dis_pix, _ = bs.get_keypoint_distances(pd_keypoint, skeleton)
            pd_dis_pix[img_idx, :] = cpd_dis_pix
            raw_info_pd[img_idx, :, :] = pd_keypoint
            img_idx += 1

    skeleton_names = cfg.skeleton_name
    if only_preds:
        save_distances_with_model_conversion(pd_dis_pix, skeleton_names, img_codes, img_apa, img_apv, os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv'), cfg.model_add)
        save_distances_zipping_by_mean(os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv'), os.path.join(args_out_img_root, f'predictions_converted_in_cm_mean.csv'))
        save_distances_zipping_by_median(os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv'), os.path.join(args_out_img_root, f'predictions_converted_in_cm_median.csv'))

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        pd_res, pd_res_latex, base_fil, pd_mae_general = comparer.generate_comparison_current(
            "predictions_converted_in_cm.csv")
        pd_res.to_csv(os.path.join(args_out_img_root, f'rm_pd_in_cm_compare.csv'), index=False)
        sel = ['code'] + skeleton_names
        base_fil = base_fil[sel]
        base_fil.to_csv(os.path.join(args_out_img_root, "rm_in_cm.csv"), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current(
            'predictions_converted_in_cm_mean.csv')
        res.to_csv(os.path.join(args_out_img_root, f'rm_pd_mean_in_cm_compare.csv'), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current(
            'predictions_converted_in_cm_median.csv')
        res.to_csv(os.path.join(args_out_img_root, f'rm_pd_median_in_cm_compare.csv'), index=False)
    else:
        save_gt_pd_compare_metrics(raw_info_gt, raw_info_pd, os.path.join(args_out_img_root, f'gt_pd_pixel_compare.csv'))

        save_distances_with_model_conversion(pd_dis_pix, skeleton_names, img_codes, img_apa, img_apv, os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv'), cfg.model_add)
        save_distances_with_model_conversion(gt_dis_pix, skeleton_names, img_codes, img_apa, img_apv, os.path.join(args_out_img_root, f'ground_truth_converted_in_cm.csv'), cfg.model_add)
        save_distances_zipping_by_mean(os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv'), os.path.join(args_out_img_root, f'predictions_converted_in_cm_mean.csv'), mask=dis_availability)
        save_distances_zipping_by_mean(os.path.join(args_out_img_root, f'ground_truth_converted_in_cm.csv'), os.path.join(args_out_img_root, f'ground_truth_converted_in_cm_mean.csv'), mask=dis_availability)
        save_distances_zipping_by_median(os.path.join(args_out_img_root, f'predictions_converted_in_cm.csv'), os.path.join(args_out_img_root, f'predictions_converted_in_cm_median.csv'), mask=dis_availability)
        save_distances_zipping_by_median(os.path.join(args_out_img_root, f'ground_truth_converted_in_cm.csv'), os.path.join(args_out_img_root, f'ground_truth_converted_in_cm_median.csv'), mask=dis_availability)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        pd_res, pd_res_latex, base_fil, pd_mae_general = comparer.generate_comparison_current("predictions_converted_in_cm.csv", mask=dis_availability)
        pd_res.to_csv(os.path.join(args_out_img_root, f'rm_pd_in_cm_compare.csv'), index=False)
        sel = ['code'] + skeleton_names
        base_fil = base_fil[sel]
        base_fil.to_csv(os.path.join(args_out_img_root, "rm_in_cm.csv"), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        gt_res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current('ground_truth_converted_in_cm.csv', mask=dis_availability)
        gt_res.to_csv(os.path.join(args_out_img_root, f'rm_gt_in_cm_compare.csv'), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current('predictions_converted_in_cm_mean.csv')
        res.to_csv(os.path.join(args_out_img_root, f'rm_pd_mean_in_cm_compare.csv'), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        gt_res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current(
            'ground_truth_converted_in_cm_mean.csv')
        gt_res.to_csv(os.path.join(args_out_img_root, f'rm_gt_mean_in_cm_compare.csv'), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current(
            'predictions_converted_in_cm_median.csv')
        res.to_csv(os.path.join(args_out_img_root, f'rm_pd_median_in_cm_compare.csv'), index=False)

        comparer = PopulationComparerByView(args_out_img_root, cfg.real_cm_data, cfg.rostrum_info)
        gt_res, gt_res_latex, base_fil, gt_mae_general = comparer.generate_comparison_current(
            'ground_truth_converted_in_cm_median.csv')
        gt_res.to_csv(os.path.join(args_out_img_root, f'rm_gt_median_in_cm_compare.csv'), index=False)
    if external_test:
        kps_info_gt = create_kps_dataframe(raw_info_gt, img_codes, img_apa, img_apv)
        kps_info_gt.to_csv(os.path.join(args_out_img_root, f'keypoints_ground_truth.csv'), index=False)
        kps_info_pd = create_kps_dataframe(raw_info_pd, img_codes, img_apa, img_apv)
        kps_info_pd.to_csv(os.path.join(args_out_img_root, f'keypoints_predictions.csv'), index=False)
        np.save(os.path.join(args_out_img_root, f'raw_pix_gt.npy'), raw_info_gt)
        np.save(os.path.join(args_out_img_root, f'raw_pix_pd.npy'), raw_info_pd)

        # Guardar las distancias convertidas por el método de referencia.
        save_distances_with_reference_model(pd_dis_pix, skeleton_names, img_codes, img_apa, img_apv, os.path.join(args_out_img_root, f'predictions_converted_in_cm_by_ref.csv'), cfg.rostrum_info)
        save_distances_zipping_by_mean(os.path.join(args_out_img_root, f'predictions_converted_in_cm_by_ref.csv'), os.path.join(args_out_img_root, f'predictions_converted_in_cm_mean_by_ref.csv'))
        save_distances_zipping_by_median(os.path.join(args_out_img_root, f'predictions_converted_in_cm_by_ref.csv'), os.path.join(args_out_img_root, f'predictions_converted_in_cm_median_by_ref.csv'))
    # if latex_tables:
    #     print(pd_res_latex)
    #     print(gt_res_latex)

    return pd_mae_general, gt_mae_general


def create_kps_dataframe(raw_info, img_codes, img_apa, img_apv):
    num_samples, num_kp, _ = raw_info.shape
    data_for_df = []
    for sample_data in raw_info:
        row = {}
        for i in range(num_kp):
            x = sample_data[i, 0]
            y = sample_data[i, 1]
            column_name = f'{i + 1}'
            row[column_name] = (x, y)
        data_for_df.append(row)
    keypoints_pd = pd.DataFrame(data_for_df)
    keypoints_pd['code'] = img_codes
    keypoints_pd['angle'] = img_apa
    keypoints_pd['point_of_view'] = img_apv
    return keypoints_pd


def save_distances_zipping_by_mean(file, out_file, mask=None):
    df = pd.read_csv(file)
    if mask is not None:
        # Poner a Nan aquellos valores que no tenemos GT disponible. Solo medidas seguras.
        df_3 = df.iloc[:, :3]
        df_16 = df.iloc[:, 3:]
        mask_bool = pd.DataFrame(mask == 1, index=df_16.index, columns=df_16.columns)
        df_16_masked = df_16.where(mask_bool, np.nan)
        df = pd.concat([df_3, df_16_masked], axis=1)
    df = df.drop('angle', axis=1)
    df['point_of_view'] = df['point_of_view'].replace(['LL', 'LR'], 'LL')
    mcl_df = df.groupby(['code', 'point_of_view']).mean(numeric_only=True).reset_index()
    mcl_df['angle'] = 0
    mcl_df.to_csv(out_file, index=False)


def save_distances_zipping_by_median(file, out_file, mask=None):
    df = pd.read_csv(file)
    if mask is not None:
        # Poner a Nan aquellos valores que no tenemos GT disponible. Solo medidas seguras.
        df_3 = df.iloc[:, :3]
        df_16 = df.iloc[:, 3:]
        mask_bool = pd.DataFrame(mask == 1, index=df_16.index, columns=df_16.columns)
        df_16_masked = df_16.where(mask_bool, np.nan)
        df = pd.concat([df_3, df_16_masked], axis=1)
    df = df.drop('angle', axis=1)
    df['point_of_view'] = df['point_of_view'].replace(['LL', 'LR'], 'LL')
    mcl_df = df.groupby(['code', 'point_of_view']).median(numeric_only=True).reset_index()
    mcl_df['angle'] = 0
    mcl_df.to_csv(out_file, index=False)


def prepare_dataframe(df, img_codes, img_apa, img_apv):
    df['code'] = img_codes
    df['angle'] = img_apa
    df['point_of_view'] = img_apv
    return df


def are_two_generations(df):
    tipo_1_pat = r'^\d+(N|R)$'
    tipo_2_pat = r'^\d+[A-Z]\d+$'

    # Comprobar si hay dos poblaciones diferentes
    has_tipo_1 = df['code'].str.match(tipo_1_pat).any()
    has_tipo_2 = df['code'].str.match(tipo_2_pat).any()
    has_total = "total" in list(df.columns)
    if has_tipo_1 and has_tipo_2 and has_total:
        return True
    else:
        return False


def generate_generation_division(df, col):
    tipo_1_pat = r'^\d+(N|R)$'
    tipo_2_pat = r'^\d+[A-Z]\d+$'

    df[f'{col}_1'] = np.where(df['code'].str.match(tipo_1_pat), df[f'{col}'], np.nan)
    df[f'{col}_2'] = np.where(df['code'].str.match(tipo_2_pat), df[f'{col}'], np.nan)

    return df


def save_distances_with_model_conversion(dis, skeleton_names, img_codes, img_apa, img_apv, out_file, config_path):
    conversor = PixelToCentimeterConversor(config_path)
    df_pix = pd.DataFrame(dis, columns=skeleton_names)
    df_pix = prepare_dataframe(df_pix, img_codes, img_apa, img_apv)
    df_cm = df_pix.copy()
    view = "lateral" if 'h_head' in skeleton_names else 'dorsal'
    if are_two_generations(df_pix):
        df_pix = generate_generation_division(df_pix, "total")
        df_pix = generate_generation_division(df_pix, "l_head")
        df_cm = generate_generation_division(df_cm, "total")
        df_cm = generate_generation_division(df_cm, "l_head")
    outliers_acc = convert_all_columns(skeleton_names, df_pix, df_cm, conversor, view)

    outliers_df = pd.concat(outliers_acc, ignore_index=True)
    outliers_df.to_csv(out_file[:-4] + "_outliers.csv", index=False)
    df = df_cm[['code', 'point_of_view', 'angle'] + skeleton_names]
    df.to_csv(out_file, index=False)


def convert_all_columns(skeleton_names, df_pix, df_cm, conversor, view):
    out = []
    for col in skeleton_names:
        if col == "w_6seg":
            print("")
        if (col == 'total' or col == 'l_head') and are_two_generations(df_pix):
            cod = col + "_1_" + view
            X = df_pix[[col + "_1"]].copy()
            X.columns = [col + "_1_pix"]
            X[col + "_1_pix"] = X[col + "_1_pix"].round(4)
            indices = list(X[X[col + "_1_pix"].isnull()].index)
            X = X.fillna(1)
            df_cm_1 = conversor.predict_by_model(cod, X)
            if len(indices) != 0:
                df_cm_1[indices] = np.nan

            cod = col + "_2_" + view
            X = df_pix[[col + "_2"]].copy()
            X.columns = [col + "_2_pix"]
            X[col + "_2_pix"] = X[col + "_2_pix"].round(4)
            indices = list(X[X[col + "_2_pix"].isnull()].index)
            X = X.fillna(1)
            df_cm_2 = conversor.predict_by_model(cod, X)
            if len(indices) != 0:
                df_cm_2[indices] = np.nan
            df_cm_f = [a if not np.isnan(a) else b for a, b in zip(df_cm_1, df_cm_2)]
            df_cm[col] = df_cm_f
            outliers = detectar_outliers(df_cm, df_pix, col)
            out.append(outliers)
        else:
            cod = col + "_" + view
            X = df_pix[[col]].copy()
            X.columns = [col + "_pix"]
            if X.isnull().all().all():
                continue
            df_cm[col] = np.array(conversor.predict_by_model(cod, X))
            outliers = detectar_outliers(df_cm, df_pix, col)
            out.append(outliers)
    return out


def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def save_distances_with_reference_model(dis, skeleton_names, img_codes, img_apa, img_apv, out_file, rostrum_info):
    shrimps_reference_method_data = "D:/1_SHRIMP_PROYECT/1_DATASET/2_ADITIONAL_INFO/shrimps_reference_method_data.json"
    shrimps_reference_method_data_json = read_json(shrimps_reference_method_data)
    date_to_factor = {}
    for image in shrimps_reference_method_data_json['images']:
        file_name = image['file_name']
        date = file_name.split('_')[-1][:-4]
        rm_id = image['id']
        ann_dict = find_dict_by_int_param(shrimps_reference_method_data_json['annotations'], 'id', rm_id)
        point_a = (ann_dict['keypoints'][0], ann_dict['keypoints'][1])
        point_b = (ann_dict['keypoints'][3], ann_dict['keypoints'][4])
        pixels_per_cm = math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
        date_to_factor[date] = pixels_per_cm

    mean_factor = sum(list(date_to_factor.values())) / len(date_to_factor)

    df_pix = pd.DataFrame(dis, columns=skeleton_names)
    df_pix = prepare_dataframe(df_pix, img_codes, img_apa, img_apv)
    df_pix['name'] = df_pix['point_of_view'].astype(str) + "_" + df_pix['angle'].astype(str) + "_" + df_pix[
        'code'].astype(str)

    df_rm = pd.DataFrame(dis, columns=skeleton_names)
    df_rm = prepare_dataframe(df_rm, img_codes, img_apa, img_apv)
    in_info = pd.read_csv(rostrum_info)
    in_info['name'] = in_info['ADDRESS'].apply(os.path.basename)
    # split in_info['name'] by _ generate columns CI, POV, AGL, BCOD
    in_info[['CI', 'POV', 'AGL', 'BCOD']] = in_info['name'].str.split('_', expand=True)
    # quitarle a in_info['BCOD'] el .png
    in_info['COD'] = in_info['BCOD'].str[:-4]
    # reconstruir un in_info['new_name'] con POV, AGL, COD
    in_info['new_name'] = in_info['POV'] + "_" + in_info['AGL'] + "_" + in_info['COD']
    in_info['date'] = in_info['ADDRESS'].str.extract(r'(\d{8})')
    checks = dict(zip(in_info['new_name'], in_info['date']))

    df_pix['date'] = df_pix['name'].map(checks)
    df_pix['factor'] = df_pix['date'].map(date_to_factor)
    df_pix['factor'] = df_pix['factor'].fillna(mean_factor)

    df_cm = pd.DataFrame(dis, columns=skeleton_names)
    df_cm = prepare_dataframe(df_cm, img_codes, img_apa, img_apv)

    outliers_acc = []

    for col in skeleton_names:
        df_cm[col] = df_pix[col] / df_pix['factor']
        outliers = detectar_outliers(df_cm, df_pix, col)
        outliers_acc.append(outliers)
    outliers_df = pd.concat(outliers_acc, ignore_index=True)
    outliers_df.to_csv(out_file[:-4] + "_outliers.csv", index=False)
    df = df_cm[['code', 'point_of_view', 'angle'] + skeleton_names]
    df.to_csv(out_file, index=False)


def find_dict_by_int_param(lis, int_key, param):
    for dic in lis:
        if param == dic.get(int_key):
            return dic
    return None


def detectar_outliers(df_cm, df_pix, columna):
    df = pd.DataFrame({
        columna: df_cm[columna],
        'pix_' + columna: df_pix[columna]
    })
    df['code'] = df_cm['code']
    df['angle'] = df_cm['angle']
    df['point_of_view'] = df_cm['point_of_view']

    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
        return pd.DataFrame(columns=list(df.columns))

    limite_inferior = Q1 - 5 * IQR  # 38
    limite_superior = Q3 + 5 * IQR  # 40

    outliers_superiores = df[df[columna] > limite_superior]
    outliers_inferiores = df[df[columna] < limite_inferior]

    outliers_finales = pd.concat([outliers_superiores, outliers_inferiores])
    outliers_finales = outliers_finales.rename(columns={columna: 'pix_val', f'pix_{columna}': 'cm_val'})
    outliers_finales.insert(0, 'variable', columna)

    df_cm.loc[df_cm[columna] > limite_superior, columna] = float('nan')
    df_cm.loc[df_cm[columna] < limite_inferior, columna] = float('nan')

    return outliers_finales


def save_gt_pd_compare_metrics(ground_truth, predictions, out_file):
    # ground_truth = pd.DataFrame(raw_info_gt, columns=keypoint_names)
    # predictions = pd.DataFrame(raw_info_pd, columns=keypoint_names)

    results_list = []

    # Calcular métricas para cada punto individualmente
    for i in range(ground_truth.shape[1]):  # Para cada punto
        y_pred = predictions[:, i, :]  # Coordenadas predichas del punto i en todas las muestras
        y_true = ground_truth[:, i, :]  # Coordenadas ground truth del punto i en todas las muestras

        mask = ~np.isnan(y_pred).any(axis=1) & ~np.isnan(y_true).any(axis=1)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        mask_to_remove = np.all(y_true == 0, axis=1)
        mask_to_keep = ~mask_to_remove
        y_true = y_true[mask_to_keep]
        y_pred = y_pred[mask_to_keep]

        # Calcular los errores
        mae = np.mean(np.abs(y_pred - y_true))  # MAE
        mse = np.mean((y_pred - y_true) ** 2)  # MSE
        rmse = np.sqrt(mse)  # RMSE
        stddev = np.std(y_pred - y_true)  # Desviación estándar
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

        distances = np.linalg.norm(y_true - y_pred, axis=1)
        epe = np.mean(distances)
        stddev_epe = np.std(distances)
        # Guardar los resultados en un diccionario
        if ground_truth.shape[1] == 23 or ground_truth.shape[1] == 37:
            num = i + 1
        elif ground_truth.shape[1] == 22:
            num = i + 2
        results_list.append({
            'Point': f'{num}',
            'MAE': mae,
            'SD(MAE)': stddev,
            'EPE': epe,
            'SD(EPE)': stddev_epe,
            'MSE': mse,
            'RMSE': rmse,
            'StdDev': stddev,
            'MAPE': mape
        })

    # Calcular métricas generales (para todos los puntos y muestras juntos)
    y_pred_total = predictions.reshape(-1, 2)  # Aplanar para tener todas las coordenadas juntas
    y_true_total = ground_truth.reshape(-1, 2)

    mask = ~np.isnan(y_pred_total).any(axis=1) & ~np.isnan(y_true_total).any(axis=1)
    y_pred_total = y_pred_total[mask]
    y_true_total = y_true_total[mask]

    mask_to_remove = np.all(y_true_total == 0, axis=1)
    mask_to_keep = ~mask_to_remove
    y_true_total = y_true_total[mask_to_keep]
    y_pred_total = y_pred_total[mask_to_keep]

    # Calcular los errores generales
    mae_total = np.mean(np.abs(y_pred_total - y_true_total))
    mse_total = np.mean((y_pred_total - y_true_total) ** 2)
    rmse_total = np.sqrt(mse_total)
    stddev_total = np.std(y_pred_total - y_true_total)
    mask = y_true_total != 0
    mape_total = np.mean(np.abs((y_true_total[mask] - y_pred_total[mask]) / y_true_total[mask]) * 100)

    distances = np.linalg.norm(y_true_total - y_pred_total, axis=1)
    epe_total = np.mean(distances)
    stddev_total_epe = np.std(distances)

    # Agregar los resultados generales a la lista
    results_list.append({
        'Point': f'General {ground_truth.shape[1]}KP',
        'MAE': mae_total,
        'SD(MAE)': stddev_total,
        'EPE': epe_total,
        'SD(EPE)': stddev_total_epe,
        'MSE': mse_total,
        'RMSE': rmse_total,
        'MAPE': mape_total
    })

    if ground_truth.shape[1] == 23:
        y_pred = predictions[:, 1:23, :]  # Coordenadas predichas del punto i en todas las muestras
        y_true = ground_truth[:, 1:23, :]
        y_pred_total = y_pred.reshape(-1, 2)  # Aplanar para tener todas las coordenadas juntas
        y_true_total = y_true.reshape(-1, 2)

        mask = ~np.isnan(y_pred_total).any(axis=1) & ~np.isnan(y_true_total).any(axis=1)
        y_pred_total = y_pred_total[mask]
        y_true_total = y_true_total[mask]

        # Calcular los errores generales
        mae_total = np.mean(np.abs(y_pred_total - y_true_total))
        mse_total = np.mean((y_pred_total - y_true_total) ** 2)
        rmse_total = np.sqrt(mse_total)
        stddev_total = np.std(y_pred_total - y_true_total)
        mape_total = np.mean(np.abs((y_true_total - y_pred_total) / y_true_total) * 100)
        distances = np.linalg.norm(y_true_total - y_pred_total, axis=1)
        epe_total = np.mean(distances)
        stddev_total_epe = np.std(distances)

        # Agregar los resultados generales a la lista
        results_list.append({
            'Point': f'General {ground_truth.shape[1] - 1}KP',
            'MAE': mae_total,
            'SD(MAE)': stddev_total,
            'EPE': epe_total,
            'SD(EPE)': stddev_total_epe,
            'MSE': mse_total,
            'RMSE': rmse_total,
            'MAPE': mape_total
        })

    # Convertir la lista de resultados en un DataFrame
    results = pd.DataFrame(results_list)
    # print(results.to_string())

    results = results.round(2)
    results = results[['Point', 'EPE', 'SD(EPE)', 'RMSE', 'MAPE']]
    results.to_csv(out_file, index=False)
    # Convertir el DataFrame a LaTeX
    latex_table = results.to_latex(index=False,
                                   caption='Error Analysis of Keypoints',
                                   label='tab:keypoints_analysis',
                                   position='hbtp',
                                   column_format='lccc',
                                   escape=False,
                                   float_format="%.2f")

    # Imprimir la tabla en formato LaTeX
    # print(latex_table)
    return results


def get_gt_image(image_np, gt_bbox, gt_keypoint, skeleton, real_info, skeleton_name, bbox_name,
                 zoom=None, args_line_width=2, args_radius=5):
    gt_image = image_np.copy()
    set_bbox_rectangle(gt_image, gt_bbox, bbox_name, color=(255, 0, 0), args_line_width=args_line_width)
    set_keypoint_joint_lines(gt_image, gt_keypoint, skeleton, skeleton_name, real_info, line_color=(0, 255, 0),
                             circle_color=(255, 0, 0))
    if zoom:
        gt_image = gt_image[zoom['y1']:zoom['y2'], zoom['x1']:zoom['x2'], :]
    return gt_image


def get_pd_image(image_np, gt_bbox, pd_keypoint, skeleton, real_info, skeleton_name, bbox_name,
                 zoom=None, args_line_width=2, args_radius=5):
    pd_image = image_np.copy()
    set_bbox_rectangle(pd_image, gt_bbox, bbox_name, color=(255, 0, 0), args_line_width=args_line_width)
    set_keypoint_joint_lines(pd_image, pd_keypoint, skeleton, skeleton_name, real_info, args_line_width=args_line_width)
    if zoom:
        pd_image = pd_image[zoom['y1']:zoom['y2'], zoom['x1']:zoom['x2'], :]
    return pd_image


def get_image_with_key_points(image_np, cfg, zoom=None):
    cp_image = image_np.copy()
    set_bbox_rectangle(cp_image, cfg['bbox'], cfg['bbox_name'], cfg['bbox_color'], cfg['bbox_line_width'],
                       cfg['bbox_text_font_scale'], cfg['bbox_text_font_thickness'])
    set_key_points_and_joints(cp_image, cfg)

    # cv2.imshow("Imagen", cp_image)
    # cv2.waitKey(0)  # Espera hasta que presiones una tecla
    # cv2.destroyAllWindows()

    if zoom:
        cp_image = cp_image[zoom['y1']:zoom['y2'], zoom['x1']:zoom['x2'], :]
    return cp_image


def set_key_points_and_joints_old(image, cfg):
    idx = 0
    cm_info = cfg['cm_info']
    for union in cfg['skeleton']:
        if union == [1, 9] or union == [2, 9]:
            idx += 1
            continue
        point1 = [int(valor) for valor in cfg['keypoints'][union[0] - 1]]
        point2 = [int(valor) for valor in cfg['keypoints'][union[1] - 1]]
        column = cfg['skeleton_names'][idx]
        joint_text = str(round(cm_info[column].values[0], 1)) if cm_info[column].values else "nan"
        image = draw_joint_with_text(image, point1, point2, joint_text, cfg['circle_color'], cfg['circle_size'],
                           cfg['joint_line_color'], cfg['joint_line_width'], cfg['joint_text_font'],
                           cfg['joint_text_font_thickness'], cfg['joint_text_font_scale'])
        idx += 1


def set_key_points_and_joints(image, cfg):
    """
    rotation_mask: Lista de 0s y 1s con la misma longitud que las uniones válidas.
                   0 = Texto paralelo a la línea (Normal).
                   1 = Texto rotado 90 grados (Perpendicular).
    """
    rotation_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    idx = 0
    cm_info = cfg['cm_info']

    # Aseguramos que la máscara tenga longitud suficiente para evitar errores
    # Si no se pasa máscara, asumimos todo 0
    if not rotation_mask:
        rotation_mask = [0] * len(cfg['skeleton'])

    for union in cfg['skeleton']:
        # Saltamos las uniones excluidas (según tu lógica original)
        if union == [1, 9] or union == [2, 9]:
            # Nota: No incrementamos idx aquí porque idx parece rastrear
            # las uniones VÁLIDAS dibujadas, no las del esqueleto total.
            # (Ajusta esto si tu idx debe coincidir con el esqueleto completo)
            idx += 1
            continue

        point1 = [int(valor) for valor in cfg['keypoints'][union[0] - 1]]
        point2 = [int(valor) for valor in cfg['keypoints'][union[1] - 1]]
        column = cfg['skeleton_names'][idx]

        joint_text = str(round(cm_info[column].values[0], 1)) if cm_info[column].values else "nan"

        # Obtenemos el valor de la máscara para esta unión actual
        # Usamos try/except o un índice seguro por si la máscara es más corta
        should_rotate_90 = False
        if idx < len(rotation_mask):
            should_rotate_90 = (rotation_mask[idx] == 1)

        image = draw_joint_with_text(
            image, point1, point2, joint_text,
            cfg['circle_color'], cfg['circle_size'],
            cfg['joint_line_color'], cfg['joint_line_width'],
            cfg['joint_text_font'], cfg['joint_text_font_thickness'],
            cfg['joint_text_font_scale'],
            rotate_90=should_rotate_90  # <--- Nuevo argumento
        )
        idx += 1
    return image


def draw_joint_with_text(img, p1, p2, joint_text, circle_color, circle_size, line_color, line_width, text_font,
                         text_thickness, text_font_scale, rotate_90=False):
    # Dibujar línea
    cv2.line(img, p1, p2, line_color, line_width)

    # Dibujar círculos
    cv2.circle(img, p1, circle_size, (0, 0, 0), -1)
    cv2.circle(img, p1, circle_size - 2, circle_color, -1)

    cv2.circle(img, p2, circle_size, (0, 0, 0), -1)
    cv2.circle(img, p2, circle_size - 2, circle_color, -1)

    # Calcular punto medio
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)

    # Calcular ángulo de la línea
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle_deg = math.degrees(angle_rad)

    # Obtener tamaño del texto
    (text_w, text_h), baseline = cv2.getTextSize(joint_text, text_font, text_font_scale, text_thickness)

    # --- LÓGICA DE ROTACIÓN ---
    # Si rotate_90 es True, sumamos 90 grados al ángulo invertido.
    # Si es False, seguimos la línea.
    # El negativo (-angle_deg) es porque el eje Y en imágenes va hacia abajo.
    rotation_angle = -angle_deg + (90 if rotate_90 else 0)

    # --- CALCULO DE OFFSET (Desplazamiento) ---
    # Ajustamos la distancia del texto a la línea.
    # Si el texto está a 90 grados, necesitamos alejarlo un poco más (mitad de su altura visual + margen)
    distancia_base = 5
    if rotate_90:
        # Si rotamos 90, el ancho del texto se convierte en altura visual
        offset_val = - (distancia_base + text_w // 2)
    else:
        offset_val = - (distancia_base + text_h)

    dx = offset_val * math.sin(angle_rad)
    dy = -offset_val * math.cos(angle_rad)  # Nota: signo cambiado para corrección perpendicular

    text_pos = (int(mid_x + dx), int(mid_y + dy))

    # --- CREAR LIENZO SEGURO (Diagonal) ---
    # Usamos la diagonal para asegurar que el texto quepa al rotar cualquier ángulo
    diag = int(math.sqrt(text_w ** 2 + text_h ** 2)) + 10
    text_img = np.zeros((diag, diag, 3), dtype=np.uint8)

    # Centrar texto en el lienzo temporal
    text_x = (diag - text_w) // 2
    text_y = (diag + text_h) // 2

    # Dibujar texto (blanco solido, sin borde negro por ahora según tu código original,
    # pero puedes usar putText dos veces si quieres borde)
    cv2.putText(text_img, joint_text, (text_x, text_y), text_font, text_font_scale, (255, 255, 255), text_thickness,
                cv2.LINE_AA)

    # Rotar el lienzo
    rot_mat = cv2.getRotationMatrix2D((diag // 2, diag // 2), rotation_angle, 1.0)
    rotated_text = cv2.warpAffine(text_img, rot_mat, (diag, diag), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # --- SUPERPOSICIÓN (Overlay) ---
    x_c, y_c = text_pos
    h_r, w_r = rotated_text.shape[:2]

    # Coordenadas top-left donde pegar
    x1 = x_c - w_r // 2
    y1 = y_c - h_r // 2
    x2 = x1 + w_r
    y2 = y1 + h_r

    # Recortes seguros
    x1_c, x2_c = max(0, x1), min(img.shape[1], x2)
    y1_c, y2_c = max(0, y1), min(img.shape[0], y2)

    if y1_c < y2_c and x1_c < x2_c:
        roi = img[y1_c:y2_c, x1_c:x2_c]

        # Recorte correspondiente en la imagen de texto rotado
        txt_roi = rotated_text[y1_c - y1: y2_c - y1, x1_c - x1: x2_c - x1]

        # Máscara simple (lo que no sea negro es texto)
        gray_txt = cv2.cvtColor(txt_roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_txt, 1, 255, cv2.THRESH_BINARY)

        # Pegar usando la máscara
        roi[mask > 0] = txt_roi[mask > 0]
        img[y1_c:y2_c, x1_c:x2_c] = roi

    return img


def draw_joint(img, p1, p2, joint_text, circle_color, circle_size, line_color, line_width, text_font, text_thickness,
               text_font_scale):
    # Dibujar línea
    cv2.line(img, p1, p2, line_color, line_width)

    cv2.circle(img, (p1[0], p1[1]), circle_size, (0, 0, 0), -1)
    cv2.circle(img, (p1[0], p1[1]), circle_size - 2, circle_color, -1)

    cv2.circle(img, (p2[0], p2[1]), circle_size, (0, 0, 0), -1)
    cv2.circle(img, (p2[0], p2[1]), circle_size - 2, circle_color, -1)
    return img


def draw_joint_with_text_old(img, p1, p2, joint_text, circle_color, circle_size, line_color, line_width, text_font,
                         text_thickness, text_font_scale):
    # Dibujar línea
    cv2.line(img, p1, p2, line_color, line_width)

    # Dibujar círculos
    cv2.circle(img, p1, circle_size, (0, 0, 0), -1)
    cv2.circle(img, p1, circle_size - 2, circle_color, -1)

    cv2.circle(img, p2, circle_size, (0, 0, 0), -1)
    cv2.circle(img, p2, circle_size - 2, circle_color, -1)

    # Calcular punto medio
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)

    # Calcular ángulo de la línea en grados
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle_deg = math.degrees(angle_rad)

    # Desplazar un poco hacia arriba (perpendicular negativa a la línea)
    offset = -15  # Puedes ajustar este valor
    dx = -offset * math.sin(angle_rad)
    dy = offset * math.cos(angle_rad)
    text_pos = (int(mid_x + dx), int(mid_y + dy))

    # Obtener tamaño del texto
    (text_w, text_h), _ = cv2.getTextSize(joint_text, text_font, text_font_scale, text_thickness)

    # Crear imagen del texto con fondo transparente
    text_img = np.zeros((text_h * 2, text_w * 2, 3), dtype=np.uint8)
    text_org = (text_w // 2, text_h + text_h // 2)
    cv2.putText(text_img, joint_text, text_org, text_font, text_font_scale, (255, 255, 255), text_thickness,
                cv2.LINE_AA)

    # Rotar la imagen del texto
    # rot_mat = cv2.getRotationMatrix2D((text_w, text_h), angle_deg, 1.0)
    rot_mat = cv2.getRotationMatrix2D((text_w, text_h), -angle_deg, 1.0)
    rotated_text = cv2.warpAffine(text_img, rot_mat, (text_img.shape[1], text_img.shape[0]), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Superponer el texto rotado en la imagen original
    x, y = text_pos
    h, w = rotated_text.shape[:2]
    x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
    x2, y2 = min(img.shape[1], x1 + w), min(img.shape[0], y1 + h)

    roi = img[y1:y2, x1:x2]
    mask = rotated_text[:y2 - y1, :x2 - x1] > 0
    roi[mask] = rotated_text[:y2 - y1, :x2 - x1][mask]
    img[y1:y2, x1:x2] = roi

    return img


def set_keypoint_joint_lines(image, keypoint, skeleton, skeleton_name, cm_info, line_color=(0, 255, 255),
                             circle_color=(0, 0, 255)):
    idx = 0
    for union in skeleton:
        point1 = [int(valor) for valor in keypoint[union[0] - 1]]
        point2 = [int(valor) for valor in keypoint[union[1] - 1]]

        column = skeleton_name[idx]
        if not cm_info[column].values:
            text = "nan"
        else:
            text = str(round(cm_info[column].values[0], 2))
        image = draw_text_on_line(image, point1, point2, text, line_color, circle_color)
        idx += 1

    idx = 0
    for union in skeleton:
        point1 = [int(valor) for valor in keypoint[union[0] - 1]]
        point2 = [int(valor) for valor in keypoint[union[1] - 1]]

        column = skeleton_name[idx]
        if not cm_info[column].values:
            text = "nan"
        else:
            text = str(round(cm_info[column].values[0], 2))
        image = draw_text_on_line_only_text(image, point1, point2, text, line_color, circle_color)
        idx += 1


def get_bounding_box(puntos_clave):
    # Extraer las coordenadas x e y ignorando los valores intermedios (en este caso los "2")
    x_coords = puntos_clave[:, 0]  # Tomar los elementos en posiciones 0, 3, 6, ...
    y_coords = puntos_clave[:, 1]  # Tomar los elementos en posiciones 1, 4, 7, ...

    # Encontrar los valores mínimo y máximo de las coordenadas x e y
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Calcular el área del bounding box
    ancho = x_max - x_min
    alto = y_max - y_min
    area = ancho * alto

    return x_min, y_min, ancho, alto


def put_rotated_text_old(img, text, pos, angle, font, scale, color, thickness):
    """
    Dibuja texto rotado en una imagen.
    """
    # 1. Calcular el tamaño del texto
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    w, h = text_size

    # 2. Crear una imagen temporal (canvas) lo suficientemente grande
    # La diagonal asegura que el texto quepa al rotar
    diag = int(np.sqrt(w ** 2 + h ** 2)) + 10
    # Imagen transparente (o negra para usar máscaras)
    txt_img = np.zeros((diag, diag, 3), dtype=np.uint8)

    # 3. Dibujar el texto centrado en el canvas
    txt_x = (diag - w) // 2
    txt_y = (diag + h) // 2
    cv2.putText(txt_img, text, (txt_x, txt_y), font, scale, color, thickness)

    # 4. Rotar el canvas
    M = cv2.getRotationMatrix2D((diag // 2, diag // 2), angle, 1.0)
    rotated_txt = cv2.warpAffine(txt_img, M, (diag, diag))

    # 5. Calcular dónde pegar el texto en la imagen original
    x_offset = pos[0] - diag // 2
    y_offset = pos[1] - diag // 2

    y1, y2 = y_offset, y_offset + diag
    x1, x2 = x_offset, x_offset + diag

    # Recortes para asegurar que no nos salimos de la imagen
    y1_c, y2_c = max(0, y1), min(img.shape[0], y2)
    x1_c, x2_c = max(0, x1), min(img.shape[1], x2)

    # Si el texto se sale completamente, no hacer nada
    if y1_c >= y2_c or x1_c >= x2_c:
        return

    # Extraer la región de interés (ROI) de la imagen original
    roi = img[y1_c:y2_c, x1_c:x2_c]

    # Extraer la región correspondiente del texto rotado
    txt_roi = rotated_txt[y1_c - y1: y2_c - y1, x1_c - x1: x2_c - x1]

    # 6. Crear máscara para pegar solo el texto (ignorar el fondo negro)
    gray_txt = cv2.cvtColor(txt_roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_txt, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Combinar fondo y texto
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    txt_fg = cv2.bitwise_and(txt_roi, txt_roi, mask=mask)
    dst = cv2.add(img_bg, txt_fg)

    # Aplicar cambios
    img[y1_c:y2_c, x1_c:x2_c] = dst


def put_rotated_text_with_outline(img, text, pos, angle, font, scale, color, thickness, outline_color=None, outline_thickness=None):
    """
    Dibuja texto rotado en una imagen, con soporte para contorno.
    """
    # Usamos el grosor del contorno (si existe) para calcular el tamaño total necesario
    th_for_size = outline_thickness if outline_thickness is not None else thickness
    text_size, baseline = cv2.getTextSize(text, font, scale, th_for_size)
    w, h = text_size

    # Lienzo temporal suficientemente grande
    diag = int(np.sqrt(w ** 2 + h ** 2)) + 20
    # Fondo negro (se usará como transparente)
    txt_img = np.zeros((diag, diag, 3), dtype=np.uint8)

    # Coordenadas para centrar el texto en el lienzo
    txt_x = (diag - w) // 2
    txt_y = (diag + h) // 2

    # --- DIBUJO EN EL LIENZO TEMPORAL ---
    # 1. Si hay contorno, se dibuja primero (más grueso)
    if outline_color is not None and outline_thickness is not None:
        # Usamos LINE_AA para bordes más suaves
        cv2.putText(txt_img, text, (txt_x, txt_y), font, scale, outline_color, outline_thickness, cv2.LINE_AA)

    # 2. Se dibuja el relleno principal (más fino) encima
    cv2.putText(txt_img, text, (txt_x, txt_y), font, scale, color, thickness, cv2.LINE_AA)
    # ------------------------------------

    # Rotar el lienzo
    M = cv2.getRotationMatrix2D((diag // 2, diag // 2), angle, 1.0)
    rotated_txt = cv2.warpAffine(txt_img, M, (diag, diag))

    # Calcular posición de pegado
    x_offset = pos[0] - diag // 2
    y_offset = pos[1] - diag // 2

    y1, y2 = y_offset, y_offset + diag
    x1, x2 = x_offset, x_offset + diag

    # Recortes de seguridad
    y1_c, y2_c = max(0, y1), min(img.shape[0], y2)
    x1_c, x2_c = max(0, x1), min(img.shape[1], x2)

    if y1_c >= y2_c or x1_c >= x2_c: return

    roi = img[y1_c:y2_c, x1_c:x2_c]
    txt_roi = rotated_txt[y1_c - y1: y2_c - y1, x1_c - x1: x2_c - x1]

    # Máscara: todo lo que no sea negro puro en el lienzo rotado se considera texto
    gray_txt = cv2.cvtColor(txt_roi, cv2.COLOR_BGR2GRAY)
    # Umbral muy bajo para capturar incluso los bordes negros del antialiasing
    _, mask = cv2.threshold(gray_txt, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    txt_fg = cv2.bitwise_and(txt_roi, txt_roi, mask=mask)
    dst = cv2.add(img_bg, txt_fg)

    img[y1_c:y2_c, x1_c:x2_c] = dst


def get_dis_err_image_not_90(image, gt_keypoint, pd_keypoint, zoom=None, args_radius=5, args_line_width=2):
    dis_err_image = image.copy()
    p_key = np.expand_dims(np.array(pd_keypoint), axis=0)
    g_key = np.expand_dims(np.array(gt_keypoint), axis=0)
    mask = np.full((p_key.shape[0], p_key.shape[1]), True)
    th = np.full((p_key.shape[0], p_key.shape[2]), 1)
    distances = _calc_distances(p_key, g_key, mask, th)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)
    for gtp, pdp, e in zip(gt_keypoint, pd_keypoint, distances):
        gtp = (int(gtp[0]), int(gtp[1]))
        pdp = (int(pdp[0]), int(pdp[1]))
        cv2.line(dis_err_image, gtp, pdp, (253, 255, 117), 1 if args_line_width - 1 < 0 else args_line_width - 1)
        cv2.circle(dis_err_image, gtp, args_radius, (255, 0, 0), -1)
        cv2.circle(dis_err_image, pdp, args_radius, (0, 0, 255), -1)

        punto_medio = (5 + (gtp[0] + pdp[0]) // 2, 5 + (gtp[1] + pdp[1]) // 2)
        cv2.putText(dis_err_image, str(round(e[0], 2)), punto_medio, font, font_scale, (0, 0, 0),
                    font_thickness + 1)
        cv2.putText(dis_err_image, str(round(e[0], 2)), punto_medio, font, font_scale, font_color,
                    font_thickness)
    if zoom:
        dis_err_image = dis_err_image[zoom['y1']:zoom['y2'], zoom['x1']:zoom['x2'], :]
    put_text_in_image(dis_err_image, "Error Distance (cm)")
    err_m = str(round(np.mean(distances), 2))
    err_me = str(round(np.median(distances), 2))
    err_d = str(round(np.std(distances), 2))
    put_text_in_image(dis_err_image, f"mean = {err_m}; median = {err_me}; std = {err_d}", pos=(10, 35), size=0.6,
                      thickness=2)
    return dis_err_image


def get_dis_err_image(image, gt_keypoint, pd_keypoint, zoom=None, args_radius=5, args_line_width=2):
    dis_err_image = image.copy()
    p_key = np.expand_dims(np.array(pd_keypoint), axis=0)
    g_key = np.expand_dims(np.array(gt_keypoint), axis=0)
    mask = np.full((p_key.shape[0], p_key.shape[1]), True)
    th = np.full((p_key.shape[0], p_key.shape[2]), 1)
    distances = _calc_distances(p_key, g_key, mask, th)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)  # Color del relleno (blanco)
    outline_color = (0, 0, 0)  # Color del borde (negro)
    # El borde debe ser sustancialmente más grueso que el relleno para que se vea
    outline_thickness = font_thickness + 3

    # Margen de separación entre la línea y el texto
    margen_separacion = 15

    for gtp, pdp, e in zip(gt_keypoint, pd_keypoint, distances):
        gtp = (int(gtp[0]), int(gtp[1]))
        pdp = (int(pdp[0]), int(pdp[1]))

        cv2.line(dis_err_image, gtp, pdp, (253, 255, 117), 1 if args_line_width - 1 < 0 else args_line_width - 1)
        cv2.circle(dis_err_image, gtp, args_radius, (255, 0, 0), -1)
        cv2.circle(dis_err_image, pdp, args_radius, (0, 0, 255), -1)

        punto_medio = ((gtp[0] + pdp[0]) // 2, (gtp[1] + pdp[1]) // 2)
        texto_error = str(round(e[0], 2))

        # Usamos el grosor del contorno para calcular el tamaño real que ocupará
        (w_text, h_text), _ = cv2.getTextSize(texto_error, font, font_scale, outline_thickness)

        # Calculamos el desplazamiento vertical para la posición (2)
        offset_y = int(w_text / 2) + margen_separacion
        punto_ajustado = (punto_medio[0], punto_medio[1] - offset_y)

        # --- LLAMADA ÚNICA A LA NUEVA FUNCIÓN ---
        put_rotated_text_with_outline(
            dis_err_image,
            texto_error,
            punto_ajustado,
            90,  # Ángulo
            font,
            font_scale,
            font_color,  # Relleno blanco
            font_thickness,  # Grosor relleno
            outline_color,  # Borde negro
            outline_thickness  # Grosor borde
        )
        # ----------------------------------------

    if zoom:
        dis_err_image = dis_err_image[zoom['y1']:zoom['y2'], zoom['x1']:zoom['x2'], :]

    put_text_in_image(dis_err_image, "Error Distance (cm)")
    err_m = str(round(np.mean(distances), 2))
    err_me = str(round(np.median(distances), 2))
    err_d = str(round(np.std(distances), 2))
    put_text_in_image(dis_err_image, f"mean = {err_m}; median = {err_me}; std = {err_d}", pos=(10, 35), size=0.6,
                      thickness=2)
    return dis_err_image


def put_text_in_image(image, text, ros_text=None, view_text=None, pos=(10, 20), size=0.7, thickness=3, font_scale=0.5):
    w = 0
    t = 0
    p = 0
    if font_scale > 0.5:
        w = 140
        t = 3
        p = 6

    if ros_text:
        draw_label(image, view_text, (0, 10), bg_color=(255, 0, 128), font_thickness=2 + t, text_color=(0, 0, 0),
                   font_scale=font_scale, padding=9 + p)
        draw_label(image, view_text, (0, 10), bg_color=(255, 0, 128), font_thickness=1 + t, text_color=(255, 255, 255),
                   font_scale=font_scale, padding=9 + p)
    if view_text:
        draw_label(image, ros_text, (130 + w, 10), bg_color=(255, 0, 255), font_thickness=2 + t, text_color=(0, 0, 0),
                   font_scale=font_scale, padding=9 + p)
        draw_label(image, ros_text, (130 + w, 10), bg_color=(255, 0, 255), font_thickness=1 + t,
                   text_color=(255, 255, 255), font_scale=font_scale, padding=9 + p)


def set_bbox_rectangle(image, bbox, text, color, line_width, text_font_scale, text_thickness):
    x, y, width, height = bbox
    fac = 30
    x = x - fac
    y = y - fac
    cv2.rectangle(image, (round(bbox[0]) - 1 - fac, round(bbox[1]) - 1 - fac),
                  (round(bbox[2]) + round(bbox[0] + fac), round(bbox[3]) + round(bbox[1]) + fac), color, line_width)

    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_font_scale,
                                                          text_thickness)
    text_bg_top_left = (x, y - text_height - 10)  # Arriba del rectángulo
    text_bg_bottom_right = (x + text_width + 10, y)
    text_bg_top_left = (int(text_bg_top_left[0]), int(text_bg_top_left[1]))
    text_bg_bottom_right = (int(text_bg_bottom_right[0]), int(text_bg_bottom_right[1]))

    overlay = image.copy()
    cv2.rectangle(overlay, text_bg_top_left, text_bg_bottom_right, color, -1)
    cv2.addWeighted(overlay, 0.9, image, 1 - 0.9, 0, image)
    text_position = (int(x + 5), int(y - 5))
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_DUPLEX, text_font_scale, (255, 255, 255), text_thickness)


def draw_label(image, text, position, text_color=(255, 255, 255), bg_color=(0, 0, 0), font_scale=0.5, font_thickness=2,
               padding=5, border_radius=8):
    # Calcula el tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
    text_width += padding * 2
    text_height += padding * 2

    # Coordenadas del rectángulo redondeado
    x, y = position
    rect_top_left = (x, y)
    rect_bottom_right = (x + text_width, y + text_height)

    # Crear una superposición para dibujar con transparencias
    overlay = image.copy()

    # Dibuja el rectángulo principal
    cv2.rectangle(overlay,
                  (x + border_radius, y),
                  (x + text_width - border_radius, y + text_height),
                  bg_color, -1)

    # Dibuja los círculos en las esquinas
    # cv2.circle(overlay, (x + border_radius, y + border_radius), border_radius, bg_color, -1)
    # cv2.circle(overlay, (x + text_width - border_radius, y + border_radius), border_radius, bg_color, -1)
    # cv2.circle(overlay, (x + border_radius, y + text_height - border_radius), border_radius, bg_color, -1)
    # cv2.circle(overlay, (x + text_width - border_radius, y + text_height - border_radius), border_radius, bg_color, -1)

    # Unir la superposición con la imagen original
    alpha = 0.9  # Controla la transparencia del fondo
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Dibuja el texto en el centro del fondo
    text_position = (x + padding, y + text_height - padding - baseline)
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thickness)


def draw_text_on_line(img, p1, p2, text, line_color=(0, 255, 0), circle_color=(255, 0, 0), text_color=(255, 255, 255),
                      thickness=2, font_scale=0.3):
    # Dibujar línea
    cv2.line(img, p1, p2, line_color, 4)
    circle_size = 10

    cv2.circle(img, (p1[0], p1[1]), circle_size, (0, 0, 0), -1)
    cv2.circle(img, (p1[0], p1[1]), circle_size - 2, circle_color, -1)
    cv2.circle(img, (p2[0], p2[1]), circle_size, (0, 0, 0), -1)
    cv2.circle(img, (p2[0], p2[1]), circle_size - 2, circle_color, -1)

    # Calcular ángulo de la línea en grados
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angulo_radianes = math.atan2(dy, dx)
    angle = math.degrees(angulo_radianes)
    if angle < 0:
        angle += 360
    angle = 360 - angle
    angle = angle % 360
    # angle = 0

    # Calcular el centro de la línea
    center_x = (p1[0] + p2[0]) // 2
    center_y = (p1[1] + p2[1]) // 2

    # Crear una imagen para el texto (transparente)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
    text_width, text_height = text_size

    # Coordenadas de origen para el texto en una imagen temporal
    text_img = np.zeros((text_height * 2, text_width * 2, 3), dtype=np.uint8)
    text_org = (text_width // 2, text_height + text_height // 2)

    # Dibujar el texto en la imagen temporal
    # cv2.putText(text_img, text, text_org, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness,
    #             lineType=cv2.LINE_AA)

    # Rotar la imagen del texto
    M = cv2.getRotationMatrix2D((text_width, text_height), angle, 1)
    rotated_text_img = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Obtener la región del texto rotado
    rows, cols, _ = rotated_text_img.shape
    overlay_x1 = max(0, center_x - cols // 2)
    overlay_y1 = max(0, center_y - rows // 2)
    overlay_x2 = min(img.shape[1], overlay_x1 + cols)
    overlay_y2 = min(img.shape[0], overlay_y1 + rows)

    # Añadir el texto rotado a la imagen principal
    overlay = rotated_text_img[:overlay_y2 - overlay_y1, :overlay_x2 - overlay_x1]
    alpha = (overlay > 0).astype(float)  # Máscara para evitar bordes negros
    img[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = (
            img[overlay_y1:overlay_y2, overlay_x1:overlay_x2] * (1 - alpha) + overlay * alpha
    ).astype(np.uint8)

    return img


def draw_text_on_line_only_text(img, p1, p2, text, line_color=(0, 255, 0), circle_color=(255, 0, 0),
                                text_color=(255, 255, 255), thickness=2, font_scale=0.8):
    """
    Dibuja una línea entre dos puntos y añade texto orientado a lo largo de la línea.
    """
    # Dibujar línea
    # cv2.line(img, p1, p2, line_color, 3)
    circle_size = 10

    # cv2.circle(img, (p1[0], p1[1]), circle_size, (0, 0, 0), -1)
    # cv2.circle(img, (p1[0], p1[1]), circle_size - 2, circle_color, -1)
    # cv2.circle(img, (p2[0], p2[1]), circle_size, (0, 0, 0), -1)
    # cv2.circle(img, (p2[0], p2[1]),  circle_size-2, circle_color, -1)

    # Calcular ángulo de la línea en grados
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angulo_radianes = math.atan2(dy, dx)
    angle = math.degrees(angulo_radianes)
    if angle < 0:
        angle += 360
    angle = 360 - angle
    angle = angle % 360
    angle = 0

    # Calcular el centro de la línea
    center_x = ((p1[0] + p2[0]) // 2)
    center_y = ((p1[1] + p2[1]) // 2)
    offset = -15
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length == 0:
        length = 1
    perp_x = -dy / length * offset
    perp_y = dx / length * offset

    # Desplazar el centro del texto por el vector perpendicular
    center_x += int(perp_x)
    center_y += int(perp_y)

    # Crear una imagen para el texto (transparente)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]  # font_scale
    text_width, text_height = text_size
    text_width = text_width

    # Coordenadas de origen para el texto en una imagen temporal
    text_img = np.zeros((text_height * 2, text_width * 2, 3), dtype=np.uint8)
    text_org = (text_width // 2, text_height + text_height // 2)

    # Dibujar el texto en la imagen temporal
    cv2.putText(text_img, text, text_org, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness,
                lineType=cv2.LINE_AA)

    # Rotar la imagen del texto
    M = cv2.getRotationMatrix2D((text_width, text_height), angle, 1)
    rotated_text_img = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Obtener la región del texto rotado
    rows, cols, _ = rotated_text_img.shape
    # rows = rows + 10
    # cols = cols + 10
    overlay_x1 = max(0, center_x - cols // 2)
    overlay_y1 = max(0, center_y - rows // 2)
    overlay_x2 = min(img.shape[1], overlay_x1 + cols)
    overlay_y2 = min(img.shape[0], overlay_y1 + rows)

    # Añadir el texto rotado a la imagen principal
    overlay = rotated_text_img[:overlay_y2 - overlay_y1, :overlay_x2 - overlay_x1]
    alpha = (overlay > 0).astype(float)  # Máscara para evitar bordes negros
    img[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = (
            img[overlay_y1:overlay_y2, overlay_x1:overlay_x2] * (1 - alpha) + overlay * alpha
    ).astype(np.uint8)

    return img


def get_image_errors(pd_keypoint, gt_keypoint, real_measure_factor):
    p_key = np.expand_dims(np.array(pd_keypoint), axis=0)
    g_key = np.expand_dims(np.array(gt_keypoint), axis=0)
    mask = np.full((p_key.shape[0], p_key.shape[1]), True)
    th = np.full((p_key.shape[0], p_key.shape[2]), 1)
    distances = _calc_distances(p_key, g_key, mask, th) / real_measure_factor
    return distances


def get_image_errors_in_pix(pd_keypoint, gt_keypoint):
    p_key = np.expand_dims(np.array(pd_keypoint), axis=0)
    g_key = np.expand_dims(np.array(gt_keypoint), axis=0)
    mask = np.full((p_key.shape[0], p_key.shape[1]), True)
    th = np.full((p_key.shape[0], p_key.shape[2]), 1)
    distances = _calc_distances(p_key, g_key, mask, th)
    return distances


def get_gt_keypoints(coco_info, image_id):
    keypoint_gt = coco_info.anns[image_id]['keypoints']
    keypoint_gt = np.array_split(keypoint_gt, len(keypoint_gt) / 3)
    keypoint_gt = np.array(keypoint_gt)
    gt_keypoint = keypoint_gt[:, :2]
    return gt_keypoint


def get_image_zoom(pd_keypoint, gt_keypoint):
    p1, p2 = get_diagonal_from_keypoints(pd_keypoint)
    p1_gt, p2_gt = get_diagonal_from_keypoints(gt_keypoint)
    zoom = {'x1': int(p1[0] if p1[0] < p1_gt[0] else p1_gt[0]),
            'y1': int(p1[1] if p1[1] < p1_gt[1] else p1_gt[1]),
            'x2': int(p2[0] if p2[0] > p2_gt[0] else p2_gt[0]),
            'y2': int(p2[1] if p2[1] > p2_gt[1] else p2_gt[1])}
    return zoom


def get_preds_for_heatmaps(heatmaps):
    preds, maxvals = _get_max_preds(heatmaps)
    N, K, H, W = heatmaps.shape
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            px = int(preds[n][k][0])
            py = int(preds[n][k][1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                preds[n][k] += np.sign(diff) * .25
    return preds[0, :, :]


def get_image_with_all_joint_heatmaps(heatmaps, real_kpts, image_data):
    heatmap_kpts = get_preds_for_heatmaps(heatmaps)
    c_h_py = heatmaps.shape[2] / 2
    c_h_px = heatmaps.shape[3] / 2
    real_htmp = np.full((image_data.shape[0], image_data.shape[1]), 0.)
    for kpt_idx in range(heatmap_kpts.shape[0]):
        if heatmap_kpts[kpt_idx][0] >= 0 and heatmap_kpts[kpt_idx][1] >= 0:
            crh_px = c_h_px - heatmap_kpts[kpt_idx][0]
            crh_py = c_h_py - heatmap_kpts[kpt_idx][1]
            cr_px = real_kpts[kpt_idx][0] + crh_px
            cr_py = real_kpts[kpt_idx][1] + crh_py
            ht = heatmaps[0, kpt_idx, :, :]
            real_htmp[round(cr_py - c_h_py):round(cr_py + c_h_py),
            round(cr_px - c_h_px):round(cr_px + c_h_px)] = real_htmp[round(cr_py - c_h_py):round(cr_py + c_h_py),
                                                           round(cr_px - c_h_px):round(cr_px + c_h_px)] + ht
    min_val = real_htmp.min()
    max_val = real_htmp.max()
    real_htmp = ((real_htmp - min_val) / (max_val - min_val)) * 255
    return real_htmp


def get_superposition_images(images_whm, img):
    sup_imgs = []
    for whm in images_whm:
        whm = whm.reshape(whm.shape[0], whm.shape[1], 1)
        whm = whm.astype(np.uint8)
        whm = cv2.applyColorMap(whm, cv2.COLORMAP_JET)
        superposicion = cv2.addWeighted(img, 0.7, whm, 0.3, 0)
        sup_imgs.append(superposicion)
        # cv2.imwrite('imagen_superpuesta.jpg', superposicion)
    return sup_imgs[0]


def get_diagonal_from_keypoints(puntos, margin=50):
    x_coords = [p[0] for p in puntos]
    y_coords = [p[1] for p in puntos]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    longitud = x_max - x_min
    ancho = y_max - y_min
    p1 = (x_min - margin if x_min - margin > 0 else 0, y_min - margin if y_min - margin > 0 else 0)
    p2 = (x_min + longitud + margin, y_min + ancho + margin)
    return p1, p2


def euclidean_distance(point_a, point_b):
    return math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
