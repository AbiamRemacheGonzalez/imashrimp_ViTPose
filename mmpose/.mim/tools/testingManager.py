from pylab import *

import argparse
import os
import os.path as osp
import warnings

import imashrimp_mmcv.mmcv as mmcv
import torch
from imashrimp_mmcv.mmcv import Config, DictAction
from imashrimp_mmcv.mmcv.runner import get_dist_info, init_dist, load_checkpoint
from custom_tools.base_tool import create_custom_file

from imashrimp_ViTPose.mmpose.datasets import build_dataloader, build_dataset
from imashrimp_ViTPose.mmpose.models import build_posenet
from imashrimp_ViTPose.mmpose.utils import setup_multi_processes

# test
from imashrimp_mmcv.mmcv.cnn import fuse_conv_bn
from imashrimp_mmcv.mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from imashrimp_ViTPose.mmpose.apis import multi_gpu_test, single_gpu_test

from custom_tools.self_labeling_engine import SelfLabeling, remove_first_image_from_cvat_backup
from custom_tools.test_tools import create_test_qualitative_images, create_test_quantitative_results
from datetime import datetime
from pixelconversor.conversor.searcher.utils import base as bs
from pixelconversor.conversor.searcher.searchers import modelSearcher
from custom_tools.self_labeling_engine import create_sample_ann_file_to_test
# from custom_tools.self_labeling_tool import SelfLabeling
# from custom_tools.self_labeling_tool import create_sample_ann_file_to_test
from custom_tools.base_tool import merge_configs

try:
    from imashrimp_mmcv.mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from imashrimp_ViTPose.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from imashrimp_ViTPose.mmpose.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--input', help='the dir of input images')
    parser.add_argument('--output', help='the dir of output images')
    parser.add_argument('--checkpoint', help='the checkpoint address')
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
    # args.config = "../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_23kp_256x192.py"
    # args.config = "../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_dorsal_23kp_256x192.py"
    args.config = "../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_large_camaron_rgbd_dorsal_23kp_256x192.py"

    # args.input = "D:/1_SHRIMP_PROYECT/1_DATASET/0_SOURCE/LangostinoDatasetLateralView/LangostinoLateralView20241120"
    args.input = "D:/1_SHRIMP_PROYECT/1_DATASET/0_SOURCE/LangostinoDatasetSuperiorView/LangostinoSuperiorView20241120"

    # args.checkpoint = "D:/vitpose_work_dir/exp_results/_23KP_EXPS/LATERAL/exp_16_0.0005_420/best_LOSS_epoch_419.pth"
    args.checkpoint = "D:/vitpose_work_dir/exp_results/_23KP_EXPS/SUPERIOR/exp_8_0.0001_210_large/best_PCK_epoch_202.pth"
    # args.checkpoint = "D:/1_SHRIMP_PROYECT/3_POSE_ESTIMATION/VITPOSE/results\lateral/23KP/experiment_3699_v0_02_07/winner_16_0.0006_423/best_LOSS_epoch_390.pth"
    # args.checkpoint = "D:/1_SHRIMP_PROYECT/3_POSE_ESTIMATION/VITPOSE/results/superior/23KP/experiment_1792_v0_02_08/winner_16_0.001_254/best_LOSS_epoch_234.pth"
    args.output = "D:/1_SHRIMP_PROYECT/1_DATASET/3_AUTO_LABELING"
    args.work_dir = "D:/1_SHRIMP_PROYECT/3_POSE_ESTIMATION/VITPOSE/results"
    args.resume_from = None
    args.gpus = None
    args.gpu_ids = None
    args.autoscale_lr = True
    args.gpu_id = 0
    args.launcher = 'none'
    args.seed = 0
    args.deterministic = False
    args.no_validate = True
    args.predict_images = True
    #
    return args


def auto_labeling():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    preapare_conversion_model(cfg, args.checkpoint)

    selfLabel = SelfLabeling(args.input, args.output)
    selfLabel.execute_self_labeling_no_test(args.config, args.checkpoint)
    selfLabel = SelfLabeling(args.input, args.output)
    selfLabel.execute_self_labeling_no_test(args.config, args.checkpoint, person_results_file=os.path.join(args.output, "test_keypoints.json"))
    # selfLabel = SelfLabeling(args.input, args.output)
    # selfLabel.execute_self_labeling_no_test(args.config, args.checkpoint, person_results_file=os.path.join(args.output, "test_keypoints.json"))
    args = parse_args()
    args.cfg_options = {}
    args.launcher = 'none'
    args.fuse_conv_bn = None
    args.gpu_id = 0
    args.tmpdir = None
    args.gpu_collect = True
    args.eval = None
    args.out = None

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    cfg_data_test = cfg.data.test
    cfg_data_test['ann_file'] = os.path.join(args.output, "test_keypoints.json")
    if "color_images" in os.listdir(args.input):
        cfg_data_test['img_prefix'] = os.path.join(args.input, "color_images")
    else:
        cfg_data_test['img_prefix'] = args.input
    if "depth_images" in os.listdir(args.input):
        cfg_data_test['img_prefix_depth'] = os.path.join(args.input, "depth_images")
    else:
        name = os.path.basename(args.input)
        base = os.path.dirname(args.input)
        base = os.path.dirname(base)
        base = os.path.join(base, "depths")
        base = os.path.join(base, name)
        cfg_data_test['img_prefix_depth'] = base
    dataset = build_dataset(cfg_data_test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
            prefetch_num=2,
            pin_memory=False,
        )),
        **dict((k, cfg.data[k]) for k in [
            'seed',
            'prefetch_num',
            'pin_memory',
            'persistent_workers',
        ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu_id])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    match = re.search(r"(.*?/results/.*/\d{2}KP)", args.checkpoint.replace("\\", "/"))
    path = match.group(1) if match else "D:/1_SHRIMP_PROYECT/3_POSE_ESTIMATION/VITPOSE/results"
    name = os.path.basename(args.input)
    # current time in string format
    fecha_hora = datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.work_dir = os.path.join(path, f"prediction_{name}_{fecha_hora}")
    checkpoint_name = os.path.basename(args.checkpoint)[:-4]
    pd_mae, gt_mae = create_test_quantitative_results(outputs, dataset, checkpoint_name, cfg, only_preds=True)
    if args.predict_images:
        create_test_qualitative_images(outputs, dataset, checkpoint_name, cfg, only_preds=True)

    results = {'mae_pd_rm': pd_mae, 'mae_gt_rm': gt_mae}
    create_custom_file(os.path.join(cfg.work_dir, "_test_quantitative_" + checkpoint_name + ".txt"), results)

    selfLabel = SelfLabeling(args.input, args.output)
    selfLabel.execute_self_labeling(outputs)
    os.remove(os.path.join(args.output, "test_keypoints.json"))


def preapare_conversion_model(cfg, checkpoint):
    match_experiment = re.search(r"(.*?/results/.*/\d{2}KP/experiment_[^/]+)", checkpoint.replace("\\", "/"))
    ruta_experiment = match_experiment.group(1) if match_experiment else None

    match_base = re.search(r"(.*?/results/(superior|lateral))", checkpoint.replace("\\", "/"))
    ruta_base = match_base.group(1) if match_base else None
    tipo_vista = match_base.group(2) if match_base else None  # "superior" o "lateral"

    if ruta_experiment and ruta_base and tipo_vista:
        nueva_carpeta = os.path.basename(ruta_experiment).replace("experiment_", "model_") + f"_{tipo_vista}"
        nueva_ruta = os.path.join(cfg.conversion_model_dir, nueva_carpeta)
        cfg.model_add = nueva_ruta + "/model_results.json"
    else:
        print("No se encontraron coincidencias en la ruta.")
        cfg.model_add = "C:/Users/Tecnico/Downloads/vitpose24102024/pixelconversor/conversor/searcher/models/model_3699_v0_02_07_lateral/model_results.json"
        cfg.model_add = "C:/Users/Tecnico/Downloads/vitpose24102024/pixelconversor/conversor/searcher/models/model_1792_v0_02_08_superior/model_results.json"


if __name__ == '__main__':
    # print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # auto_labeling()
    remove_first_image_from_cvat_backup("C:/Users/Tecnico/Downloads/superior_2024_06_01_4.zip")
