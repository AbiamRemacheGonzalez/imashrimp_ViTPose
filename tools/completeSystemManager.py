from mmcv import Config, DictAction
from custom_tools.training_engine import TrainingEngine
import argparse
import os
import sys

# Ruta al proyecto externo
ruta_proyecto = os.path.abspath('D:/1_SHRIMP_PROYECT/4_CLASSIFICATION/BINARY_CLASSIFICATION')
ruta_proyecto_1 = os.path.abspath('D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT')
# Agregar al sys.path
if ruta_proyecto not in sys.path:
    sys.path.append(ruta_proyecto)

if ruta_proyecto_1 not in sys.path:
    sys.path.append(ruta_proyecto_1)

# Ahora puedes importar módulos desde ese proyecto
from classification_manager.managers.BinaryClassificationManager import BinaryClassificationManager
from DATASETMANAGEMENT.managers.managers.manager_for_complete_system.ShrimpDatasetManagerForCompleteSystem import ShrimpDatasetManagerForCompleteSystem


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


if __name__ == '__main__':
    general_config = "C:/Users/Tecnico/Downloads/vitpose24102024/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/CompleteSystemConfig.py"
    general_cfg = Config.fromfile(general_config)
    pose_dir = general_cfg.complete_system_config.complete_system_data_root + "/pose_estimation"

    # classification_dir = general_cfg.complete_system_config.complete_system_data_root + "/classification"
    # point_of_view_classification, pov_fails = BinaryClassificationManager.predict(classification_dir, "point_of_view", general_cfg.complete_system_config.point_of_view_pth, general_cfg.complete_system_config.results_dir)
    # print(pov_fails)
    # pov_fails.to_csv(general_cfg.complete_system_config.results_dir + "/tests_after_classification/pov_fails.csv", index=False)
    # point_of_view_classification.columns = ["ADDRESS", "POINT_OF_VIEW"]
    # rostrum_classification, ri_fails = BinaryClassificationManager.predict(classification_dir, "rostrum_integrity", general_cfg.complete_system_config.rostrum_integrity_pth, general_cfg.complete_system_config.results_dir)
    # print(ri_fails)
    # ri_fails.to_csv(general_cfg.complete_system_config.results_dir + "/tests_after_classification/ri_fails.csv", index=False)
    # rostrum_classification.columns = ["ADDRESS", "ROSTRUM_INTEGRITY"]
    # #temporal
    # rostrum_classification['ROSTRUM_INTEGRITY'] = 1 - rostrum_classification['ROSTRUM_INTEGRITY']
    # #
    # complete_classification = point_of_view_classification.merge(rostrum_classification, on="ADDRESS")
    #
    # res_dir = general_cfg.complete_system_config.results_dir + "/tests_after_classification"
    # ShrimpDatasetManagerForCompleteSystem.create_test_dataset_for_pose_estimation(complete_classification, pose_dir, res_dir)
    pose_data_roots = os.listdir(pose_dir)
    for data_root in pose_data_roots:
        splits = data_root.split("_")
        view = splits[3]
        nkp = splits[2][:-2]
        data_root = os.path.join(general_cfg.complete_system_config.complete_system_data_root, "pose_estimation", data_root)
        config = general_cfg.complete_system_config.networks[f"{view}_{nkp}"][0]
        pretrain = general_cfg.complete_system_config.networks[f"{view}_{nkp}"][1]

        cfg = Config.fromfile(config)
        cfg.data_root = data_root
        cfg.ann_file_measure = f'{data_root}/annotations/real_measure.json'
        cfg.results_dir = general_cfg.complete_system_config.results_dir
        cfg.data.train.ann_file = f'{data_root}/annotations/train_keypoints.json'
        cfg.data.train.img_prefix = f'{data_root}/images/train/'
        cfg.data.train.img_prefix_depth = f'{data_root}/depths/train/'
        cfg.data.val.ann_file = f'{data_root}/annotations/val_keypoints.json'
        cfg.data.val.img_prefix = f'{data_root}/images/val/'
        cfg.data.val.img_prefix_depth = f'{data_root}/depths/val/'
        cfg.data.test.ann_file = f'{data_root}/annotations/test_keypoints.json'
        cfg.data.test.img_prefix = f'{data_root}/images/test/'
        cfg.data.test.img_prefix_depth = f'{data_root}/depths/test/'

        args = parse_args(config, pretrain)
        TrainingEngine(args, cfg)()
