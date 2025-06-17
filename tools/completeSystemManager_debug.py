from mmcv import Config, DictAction
from custom_tools.training_engine import TrainingEngine
import argparse
import os
import pandas as pd
import sys
import numpy as np
import glob
import custom_tools.test_tools as test_tools
from custom_tools.base_tool import create_dir
from pixelconversor.conversor.comparer.comparer import PopulationComparerByView

ruta_proyecto = os.path.abspath('D:/1_SHRIMP_PROYECT/4_CLASSIFICATION/BINARY_CLASSIFICATION')
ruta_proyecto_1 = os.path.abspath('D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT')
if ruta_proyecto not in sys.path:
    sys.path.append(ruta_proyecto)

if ruta_proyecto_1 not in sys.path:
    sys.path.append(ruta_proyecto_1)

# Ahora puedes importar módulos desde ese proyecto
from classification_manager.managers.BinaryClassificationManager import BinaryClassificationManager
from DATASETMANAGEMENT.managers.managers.manager_for_complete_system.ShrimpDatasetManagerForCompleteSystem import \
    ShrimpDatasetManagerForCompleteSystem

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args(config_file, pretrained_model):
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--predict_images', type=bool, default=True)
    # debug
    args = parser.parse_args(args=[
        "../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron"
        "/VitPose_small_camaron_rgbd_superior_multiple_256x192.py"])
    # args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # debug
    args.config = config_file
    args.cfg_options = {"model.pretrained": pretrained_model}
    args.resume_from = None
    args.gpus = None
    args.gpu_ids = None
    args.autoscale_lr = True
    args.gpu_id = 0
    args.launcher = 'none'
    args.seed = 0
    args.deterministic = False
    args.no_validate = True
    #
    return args


def analysis_of_predictions(data_dirs, res_dir, cfg, mode=""):
    lateral_pred, dorsal_pred = get_pred_info_joining_by_view(data_dirs, mode)

    name = "comparison_preds_vs_real" if mode == "" else f"comparison_{mode}_preds_vs_real"
    cm_dir = os.path.join(res_dir, name)
    create_dir(cm_dir)
    selected_pov = get_best_point_of_view(lateral_pred, dorsal_pred, cm_dir, cfg)
    in_cm_dir = os.path.join(cm_dir, "_preds_and_real_files")
    create_dir(in_cm_dir)
    if selected_pov == "lateral":
        pred_name = f'1_predictions_by_lateral.csv'
        cols_diferentes = [col for col in dorsal_pred.columns if col not in lateral_pred.columns]
        pred_by_lateral = lateral_pred.merge(dorsal_pred[['code', 'point_of_view', 'angle'] + cols_diferentes],
                                             on=['code', 'point_of_view', 'angle'], how='outer')
        pred_by_lateral.to_csv(os.path.join(in_cm_dir, pred_name), index=False)
    elif selected_pov == "dorsal":
        pred_name = f'1_predictions_by_dorsal.csv'
        cols_diferentes = [col for col in lateral_pred.columns if col not in dorsal_pred.columns]
        pred_by_dorsal = dorsal_pred.merge(lateral_pred[['code', 'point_of_view', 'angle'] + cols_diferentes],
                                           on=['code', 'point_of_view', 'angle'], how='outer')
        pred_by_dorsal.to_csv(os.path.join(in_cm_dir, pred_name), index=False)

    comparer = PopulationComparerByView(in_cm_dir, cfg.real_cm_data, cfg.rostrum_info)
    pd_res, pd_res_latex, base_fil, pd_mae_general = comparer.generate_comparison_current(pred_name)
    pd_res.to_csv(os.path.join(cm_dir, f'5_rm_pd_in_cm_compare_{selected_pov}.csv'), index=False)
    base_fil.to_csv(os.path.join(in_cm_dir, f"1_rm_in_cm_{selected_pov}.csv"), index=False)


def get_best_point_of_view(lateral_df, dorsal_df, res_dir, cfg):
    best_dir = os.path.join(res_dir, "_lateral_vs_dorsal_lengths")
    create_dir(best_dir)
    lateral_codes = list(set(list(lateral_df['code'].unique())))
    dorsal_codes = list(set(list(dorsal_df['code'].unique())))
    same_codes = list(set(lateral_codes) & set(dorsal_codes))
    lateral_preds_sub = lateral_df[lateral_df['code'].isin(same_codes)]
    lateral_preds_sub.to_csv(os.path.join(best_dir, f'_lateral_lenghts_subdf.csv'), index=False)
    dorsal_preds_sub = dorsal_df[dorsal_df['code'].isin(same_codes)]
    dorsal_preds_sub.to_csv(os.path.join(best_dir, f'_dorsal_lenghts_subdf.csv'), index=False)

    comparer = PopulationComparerByView(best_dir, cfg.real_cm_data, cfg.rostrum_info)
    pd_res_lat, pd_res_latex, base_fil, pd_mae_general = comparer.generate_comparison_current("_lateral_lenghts_subdf.csv")
    pd_res_lat = pd_res_lat.head(10)
    valor_mae_lat = pd_res_lat.loc[pd_res_lat['Variable'] == 'Lengths', 'MAE'].values[0]

    comparer = PopulationComparerByView(best_dir, cfg.real_cm_data, cfg.rostrum_info)
    pd_res_dor, pd_res_latex, base_fil, pd_mae_general = comparer.generate_comparison_current("_dorsal_lenghts_subdf.csv")
    pd_res_dor = pd_res_dor.head(10)
    valor_mae_dor = pd_res_dor.loc[pd_res_dor['Variable'] == 'Lengths', 'MAE'].values[0]

    pd_res_dor = pd_res_dor.applymap(lambda x: f'{x:.3f}' if isinstance(x, float) else x)
    pd_res_lat = pd_res_lat.applymap(lambda x: f'{x:.3f}' if isinstance(x, float) else x)

    pd_res_dor['MAE'] = pd_res_dor['MAE'] + ' ±' + pd_res_dor['StdDev']
    pd_res_dor = pd_res_dor.drop(columns=['StdDev'])
    pd_res_dor = pd_res_dor[['Variable', 'MAE', 'RMSE', 'MAPE']]
    pd_res_dor = pd.concat([pd_res_dor.loc[pd_res_dor['Variable'] == 'total'], pd_res_dor.loc[pd_res_dor['Variable'] != 'total']]).reset_index(drop=True)
    pd_res_dor['Variable'] = pd_res_dor['Variable'].str.replace('_', '\\_')

    pd_res_lat['MAE'] = pd_res_lat['MAE'] + ' ±' + pd_res_lat['StdDev']
    pd_res_lat = pd_res_lat.drop(columns=['StdDev'])
    pd_res_lat = pd_res_lat[['Variable', 'MAE', 'RMSE', 'MAPE']]
    pd_res_lat = pd.concat([pd_res_lat.loc[pd_res_lat['Variable'] == 'total'], pd_res_lat.loc[pd_res_lat['Variable'] != 'total']]).reset_index(drop=True)
    pd_res_lat['Variable'] = pd_res_lat['Variable'].str.replace('_', '\\_')

    # Variable MAE(cm) RMSE(cm) MAPE(%)
    if valor_mae_lat < valor_mae_dor:
        pd_res_lat.iloc[-1] = pd_res_lat.iloc[-1].apply(lambda x: f'\\textbf{{{x}}}')
    else:
        pd_res_dor.iloc[-1] = pd_res_dor.iloc[-1].apply(lambda x: f'\\textbf{{{x}}}')
    tabla = pd.concat([pd_res_lat, pd_res_dor], axis=1, keys=['Lateral Point of View', 'Dorsal Point of View'])
    latex = tabla.to_latex(index=False, escape=False, column_format='lccc|lccc')

    with open(res_dir + '/4_lateral_vs_dorsal_lengths.tex', 'w') as f:
        f.write(latex)

    return "lateral" if valor_mae_lat < valor_mae_dor else "dorsal"


def get_pred_info_joining_by_view(data_dirs, mode=""):
    lateral = None
    dorsal = None
    for key, value in data_dirs.items():
        view = key.split("_")[0]
        main_dir = data_dirs[key]
        name = "predictions_converted_in_cm.csv" if mode == "" else f"predictions_converted_in_cm_{mode}.csv"
        predictions_converted_in_cm = os.path.join(main_dir, name)
        predictions_converted_in_cm_df = pd.read_csv(predictions_converted_in_cm)
        if view == "lateral":
            if lateral is None:
                lateral = predictions_converted_in_cm_df
            else:
                lateral = pd.concat([lateral, predictions_converted_in_cm_df], ignore_index=True)

        if view == "dorsal":
            if dorsal is None:
                dorsal = predictions_converted_in_cm_df
            else:
                dorsal = pd.concat([dorsal, predictions_converted_in_cm_df], ignore_index=True)

    return lateral, dorsal


def get_pixel_info_joining_by_view(data_dirs):
    lateral_pix_pred = None
    lateral_pix_gt = None
    dorsal_pix_pred = None
    dorsal_pix_gt = None

    for key, value in data_dirs.items():
        view = key.split("_")[0]
        nkp = int(key.split("_")[1])
        main_dir = data_dirs[key]

        pred_pixel = os.path.join(main_dir, "raw_pix_pd.npy")
        pred_pixel_arr = np.load(pred_pixel)

        gt_pixel = os.path.join(main_dir, "raw_pix_gt.npy")
        gt_pixel_arr = np.load(gt_pixel)

        if view == "lateral":
            if nkp == 22:
                pred_pixel_arr = np.pad(pred_pixel_arr, ((0, 0), (1, 0), (0, 0)), constant_values=np.nan)
                gt_pixel_arr = np.pad(gt_pixel_arr, ((0, 0), (1, 0), (0, 0)), constant_values=np.nan)

            if lateral_pix_pred is None:
                lateral_pix_pred = pred_pixel_arr
            else:
                lateral_pix_pred = np.concatenate((lateral_pix_pred, pred_pixel_arr), axis=0)

            if lateral_pix_gt is None:
                lateral_pix_gt = gt_pixel_arr
            else:
                lateral_pix_gt = np.concatenate((lateral_pix_gt, gt_pixel_arr), axis=0)

        if view == "dorsal":
            if nkp == 22:
                pred_pixel_arr = np.pad(pred_pixel_arr, ((0, 0), (1, 0), (0, 0)), constant_values=np.nan)
                gt_pixel_arr = np.pad(gt_pixel_arr, ((0, 0), (1, 0), (0, 0)), constant_values=np.nan)

            if dorsal_pix_pred is None:
                dorsal_pix_pred = pred_pixel_arr
            else:
                dorsal_pix_pred = np.concatenate((dorsal_pix_pred, pred_pixel_arr), axis=0)

            if dorsal_pix_gt is None:
                dorsal_pix_gt = gt_pixel_arr
            else:
                dorsal_pix_gt = np.concatenate((dorsal_pix_gt, gt_pixel_arr), axis=0)

    return lateral_pix_pred, lateral_pix_gt, dorsal_pix_pred, dorsal_pix_gt


def join_comparison_results(res_dir, compare_1="", compare_2=""):
    name_1 = "comparison_preds_vs_real" if compare_1 == "" else f"comparison_{compare_1}_preds_vs_real"
    cm_dir_1 = os.path.join(res_dir, name_1)

    name_2 = "comparison_preds_vs_real" if compare_2 == "" else f"comparison_{compare_2}_preds_vs_real"
    cm_dir_2 = os.path.join(res_dir, name_2)

    compare_1_dir = os.path.join(cm_dir_1, f'5_rm_pd_in_cm_compare_lateral.csv')
    if not os.path.exists(compare_1_dir):
        compare_1_dir = os.path.join(cm_dir_1, f'5_rm_pd_in_cm_compare_dorsal.csv')

    compare_2_dir = os.path.join(cm_dir_2, f'5_rm_pd_in_cm_compare_lateral.csv')
    if not os.path.exists(compare_2):
        compare_2_dir = os.path.join(cm_dir_2, f'5_rm_pd_in_cm_compare_dorsal.csv')

    compare_1_df = pd.read_csv(compare_1_dir)
    compare_2_df = pd.read_csv(compare_2_dir)

    compare_1_df = compare_1_df[['Variable', 'MAE', 'StdDev', 'RMSE', 'MAPE']]
    compare_2_df = compare_2_df[['Variable', 'MAE', 'StdDev', 'RMSE', 'MAPE']]

    compare_1_df = compare_1_df.round(2).astype(str)
    compare_2_df = compare_2_df.round(2).astype(str)

    compare_1_df['MAE'] = compare_1_df['MAE'] + ' ± ' + compare_1_df['StdDev']
    compare_2_df['MAE'] = compare_2_df['MAE'] + ' ± ' + compare_2_df['StdDev']

    compare_1_df = compare_1_df.drop(columns=['StdDev'])
    compare_2_df = compare_2_df.drop(columns=['StdDev'])

    compare_1_df = pd.concat([compare_1_df.loc[compare_1_df['Variable'] == 'total'], compare_1_df.loc[compare_1_df['Variable'] != 'total']]).reset_index(drop=True)
    compare_2_df = pd.concat([compare_2_df.loc[compare_2_df['Variable'] == 'total'], compare_2_df.loc[compare_2_df['Variable'] != 'total']]).reset_index(drop=True)

    compare_1_df = compare_1_df.iloc[:-1]
    compare_2_df = compare_2_df.iloc[:-1]

    compare_1_df['Variable'] = compare_1_df['Variable'].str.replace('_', '\\_')
    compare_2_df['Variable'] = compare_2_df['Variable'].str.replace('_', '\\_')

    combined_df = pd.concat([compare_2_df.reset_index(drop=True), compare_1_df.reset_index(drop=True)], axis=1,
                            keys=['Conversion not using regression', 'Conversion using regression'])
    latex = combined_df.to_latex(index=False, escape=False, column_format='lccc|lccc')

    name = f'5_not_reg_vs_reg.tex' if compare_1 == "" else f'5_not_reg_vs_reg_{compare_1}.tex'
    with open(res_dir + f'/{name}', 'w') as f:
        f.write(latex)


def rostrum_discrimination(general_cfg):
    main_res_dir = general_cfg.complete_system_config.results_dir + "/1_classification/2_rostrum_integrity"
    create_dir(main_res_dir)
    name = main_res_dir + "/_ri_info.csv"
    if os.path.exists(name):
        return pd.read_csv(name)
    classification_dir = general_cfg.complete_system_config.complete_system_data_root + "/classification"
    rostrum_classification, ri_fails = BinaryClassificationManager.predict(classification_dir, "rostrum_integrity",
                                                                           general_cfg.complete_system_config.rostrum_integrity_pth,
                                                                           main_res_dir)
    ri_fails.to_csv(main_res_dir + "/ri_fails.csv", index=False)
    rostrum_classification.columns = ["ADDRESS", "ROSTRUM_INTEGRITY"]
    # temporal
    # rostrum_classification['ROSTRUM_INTEGRITY'] = 1 - rostrum_classification['ROSTRUM_INTEGRITY']
    #
    rostrum_classification.to_csv(name, index=False)
    return rostrum_classification


def point_of_view_discrimination(general_cfg):
    main_res_dir = general_cfg.complete_system_config.results_dir + "/1_classification/1_point_of_view"
    create_dir(main_res_dir)
    name = main_res_dir + "/_pov_info.csv"
    if os.path.exists(name):
        return pd.read_csv(name)
    classification_dir = general_cfg.complete_system_config.complete_system_data_root + "/classification"
    point_of_view_classification, pov_fails = BinaryClassificationManager.predict(classification_dir, "point_of_view", general_cfg.complete_system_config.point_of_view_pth, main_res_dir)
    pov_fails.to_csv(main_res_dir + "/pov_fails.csv", index=False)
    point_of_view_classification.columns = ["ADDRESS", "POINT_OF_VIEW"]
    point_of_view_classification.to_csv(name, index=False)

    return point_of_view_classification


def create_test_dataset_for_pose_estimation(point_of_view_classification, rostrum_classification, general_cfg):
    main_res_dir = general_cfg.complete_system_config.results_dir + "/1_classification/3_dataset_for_pose_estimation"
    pose_dir = general_cfg.complete_system_config.complete_system_data_root + "/pose_estimation"
    complete_classification = point_of_view_classification.merge(rostrum_classification, on="ADDRESS")
    ShrimpDatasetManagerForCompleteSystem.create_test_dataset_for_pose_estimation(complete_classification, pose_dir, main_res_dir)
    return main_res_dir


def pose_estimation(general_cfg):
    pose_dir = general_cfg.complete_system_config.complete_system_data_root + "/pose_estimation"
    pose_data_roots = os.listdir(pose_dir)
    data_dirs = {}
    for data_root in pose_data_roots:
        splits = data_root.split("_")
        view = splits[3]
        nkp = splits[2][:-2]
        data_root = os.path.join(general_cfg.complete_system_config.complete_system_data_root, "pose_estimation",
                                 data_root)
        config = general_cfg.complete_system_config.networks[f"{view}_{nkp}"][0]
        pretrain = general_cfg.complete_system_config.networks[f"{view}_{nkp}"][1]

        cfg = Config.fromfile(config)
        cfg.data_root = data_root
        cfg.ann_file_measure = f'{data_root}/annotations/real_measure.json'
        cfg.results_dir = general_cfg.complete_system_config.results_dir + f"/2_pose_estimation/1_train"
        cfg.data.train.ann_file = f'{data_root}/annotations/train_keypoints.json'
        cfg.data.train.img_prefix = f'{data_root}/images/train/'
        cfg.data.train.img_prefix_depth = f'{data_root}/depths/train/'
        cfg.data.val.ann_file = f'{data_root}/annotations/val_keypoints.json'
        cfg.data.val.img_prefix = f'{data_root}/images/val/'
        cfg.data.val.img_prefix_depth = f'{data_root}/depths/val/'
        cfg.data.test.ann_file = f'{data_root}/annotations/test_keypoints.json'
        cfg.data.test.img_prefix = f'{data_root}/images/test/'
        cfg.data.test.img_prefix_depth = f'{data_root}/depths/test/'
        cfg.conversion_model_dir = general_cfg.complete_system_config.results_dir + "/3_size_estimation"
        create_dir(cfg.conversion_model_dir)

        args = parse_args(config, pretrain)
        TrainingEngine(args, cfg)()
        dataset_for_pose_estimation = general_cfg.complete_system_config.results_dir + "/1_classification/3_dataset_for_pose_estimation"
        result_for_pose_estimation = general_cfg.complete_system_config.results_dir + f"/2_pose_estimation/2_test"
        create_dir(result_for_pose_estimation)
        data_dir = TrainingEngine(args, cfg).external_test(dataset_for_pose_estimation, result_for_pose_estimation, view, nkp)
        data_dirs[f"{view}_{nkp}"] = data_dir
    return data_dirs


def join_complete_system_results(data_dirs, general_cfg):
    res_dir = general_cfg.complete_system_config.results_dir + f"/4_final_results"
    create_dir(res_dir)

    # 1 Classification
    get_discriminators_results(res_dir)
    # 2 Pose Estimation
    get_pose_estimation_results(data_dirs, res_dir)
    # 3 Size Regression
    get_size_regression_results(data_dirs, res_dir, general_cfg)


def get_size_regression_results(data_dirs, res_dir, general_cfg):
    res_dir = res_dir + "/3_size_regression"
    create_dir(res_dir)
    config = general_cfg.complete_system_config.networks[f"lateral_23"][0]
    cfg = Config.fromfile(config)
    analysis_of_predictions(data_dirs, res_dir, cfg)
    analysis_of_predictions(data_dirs, res_dir, cfg, mode="mean")
    analysis_of_predictions(data_dirs, res_dir, cfg, mode="median")

    analysis_of_predictions(data_dirs, res_dir, cfg, mode="by_ref")
    analysis_of_predictions(data_dirs, res_dir, cfg, mode="mean_by_ref")
    analysis_of_predictions(data_dirs, res_dir, cfg, mode="median_by_ref")

    join_comparison_results(res_dir, compare_1="", compare_2="by_ref")
    join_comparison_results(res_dir, compare_1="mean", compare_2="mean_by_ref")
    join_comparison_results(res_dir, compare_1="median", compare_2="median_by_ref")


def get_discriminators_results(res_dir):
    main_dir = os.path.dirname(res_dir) + "/1_classification"
    create_dir(main_dir)
    pov = main_dir + "/1_point_of_view"
    for root, dirs, files in os.walk(pov):
        if 'results.csv' in files:
            pov = os.path.join(root, 'results.csv')
    pov_df = pd.read_csv(pov)
    ros = main_dir + "/2_rostrum_integrity"
    for root, dirs, files in os.walk(ros):
        if 'results.csv' in files:
            ros = os.path.join(root, 'results.csv')
    ros_df = pd.read_csv(ros)
    pov_df = pov_df.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)
    ros_df = ros_df.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)
    concatenated = pd.concat([pov_df, ros_df], axis=1, keys=['Point of View', 'Rostrum Integrity'])
    concatenated.iloc[-1] = concatenated.iloc[-1].apply(lambda x: f'\\textbf{{{x}}}')
    latex = concatenated.to_latex(index=False, escape=False, column_format='lcc|lcc')
    res_dir = res_dir + "/1_classification"
    create_dir(res_dir)
    add = res_dir + "/1_pov_and_ros_results.tex"
    with open(add, 'w') as f:
        f.write(latex)


def get_pose_estimation_results(data_dirs, res_dir):
    res_dir = res_dir + "/2_pose_estimation"
    create_dir(res_dir)
    analysis_of_key_points(data_dirs, res_dir)
    info = []
    for key, value in data_dirs.items():
        name = os.path.basename(value) + ".txt"
        name = name.replace("qualitative", "quantitative")
        base = os.path.dirname(value)
        path = os.path.join(base, name)
        AP = get_value_from_file(path, "AP")
        PCK10 = get_value_from_file(path, "PCKe_10")
        ni = count_images_in_folder(os.path.join(value, "combined"))
        info.append({
            "Model": key,
            "Nº Images": ni,
            "mAP 50:95(%)": AP,
            "PCK_10px(%)": PCK10
        })
    info.append({
        "Model": "General",
        "Nº Images": sum([i["Nº Images"] for i in info]),
        "mAP 50:95(%)": sum([i["mAP 50:95(%)"] for i in info]) / len(info),
        "PCK_10px(%)": sum([i["PCK_10px(%)"] for i in info]) / len(info)})
    df_info = pd.DataFrame(info)
    df_info = df_info.applymap(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
    df_info.iloc[-1] = df_info.iloc[-1].apply(lambda x: f'\\textbf{{{x}}}')
    # To latex
    latex = df_info.to_latex(index=False, escape=False, column_format='|c|c|c|c|')
    add = res_dir + "/2_neural_networks_results.tex"
    create_dir(res_dir)
    with open(add, 'w') as f:
        f.write(latex)


def get_value_from_file(file_path, key):
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                k, v = line.split(':', 1)
                k = k.strip()
                v = v.strip()
                if k == key:
                    try:
                        return float(v)
                    except ValueError:
                        raise ValueError(f"El valor '{v}' para la clave '{k}' no es un float válido.")
    raise KeyError(f"La clave '{key}' no fue encontrada en el archivo.")


def count_images_in_folder(folder_path):
    # Extensiones de imágenes comunes
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    count = 0
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in image_extensions:
            count += 1
    return count


def analysis_of_key_points(data_dirs, res_dir):
    lateral_pix_pred, lateral_pix_gt, dorsal_pix_pred, dorsal_pix_gt = get_pixel_info_joining_by_view(data_dirs)
    pixels_dir = os.path.join(res_dir, "_pixel_comparison")
    create_dir(pixels_dir)

    test_tools.save_gt_pd_compare_metrics(lateral_pix_gt, lateral_pix_pred,
                                          pixels_dir + "/gt_pd_pixel_compare_lateral.csv")
    test_tools.save_gt_pd_compare_metrics(dorsal_pix_gt, dorsal_pix_pred,
                                          pixels_dir + "/gt_pd_pixel_compare_dorsal.csv")

    pixel_df_lateral = pd.read_csv(pixels_dir + "/gt_pd_pixel_compare_lateral.csv")
    pixel_df_dorsal = pd.read_csv(pixels_dir + "/gt_pd_pixel_compare_dorsal.csv")
    pixel_df_lateral = pixel_df_lateral.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)
    pixel_df_dorsal = pixel_df_dorsal.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)

    tabla = pd.concat([pixel_df_lateral, pixel_df_dorsal], axis=1,
                      keys=['Lateral Point of View', 'Dorsal Point of View'])
    tabla = tabla.iloc[:-1]
    tabla.iloc[-1] = tabla.iloc[-1].apply(lambda x: f'\\textbf{{{x}}}')
    tabla_latex = tabla.to_latex(index=False, escape=False, column_format='c' * 8)
    with open(res_dir + '/3_gt_pd_pixel_compare.tex', 'w') as f:
        f.write(tabla_latex)


if __name__ == '__main__':
    general_config = "C:/Users/Tecnico/Downloads/vitpose24102024/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/CompleteSystemConfig.py"
    general_cfg = Config.fromfile(general_config)
    name = os.path.basename(general_cfg.complete_system_config.complete_system_data_root)
    name = name.replace("shrimp_dataset_", "")
    general_cfg.complete_system_config.results_dir = general_cfg.complete_system_config.results_dir + f"/{name}"
    create_dir(general_cfg.complete_system_config.results_dir)

    # 1. Classification of test images and dataset creation for pose estimation.
    # 1.1 Point of view classification of test images.
    point_of_view_classification = point_of_view_discrimination(general_cfg)
    # 1.2 Rostrum integrity classification of test images.
    rostrum_classification = rostrum_discrimination(general_cfg)
    # 1.3 Create test dataset using classification results for pose estimation.
    create_test_dataset_for_pose_estimation(point_of_view_classification, rostrum_classification, general_cfg)

    # 2. Pose Estimation of test dataset result of dopple classification.
    data_dirs = pose_estimation(general_cfg)

    # 3. Joining results of classification and pose estimation.
    join_complete_system_results(data_dirs, general_cfg)

    # Cosas a mejorar:
    # 1. Ajustar las redes de clasificación. Tienen un menor desempeño que las del paper anteriormente.
    # 2. Estoy convencido de que no influirá en mucho. Añadir el total y abdomen como una suma y no como un valor punto a punto.
    # 3. La red que discrimina rostrum está actualmente inversa. Debe ser corregida.

    # 1. Añadir el redimiento de las redes por separado en las imágenes de test. (hecho)
    # 2. Usar las redes lateral 22 y 23 entrenadas con más épocas. (hecho)
    # 3. Incluir los valores de calculados con el método de referencia. El que usa una conversión directa entre una regla y el valor en píxeles. (hecho)
