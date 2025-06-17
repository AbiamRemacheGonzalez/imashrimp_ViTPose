# Copyright (c) OpenMMLab. All rights reserved.
from pylab import *
import json
import copy
import time
from mmcv.runner import set_random_seed
from mmcv.utils import get_git_hash

from mmpose import __version__
from mmpose.apis import init_random_seed, train_model
from mmpose.utils import collect_env, get_root_logger
import argparse
import os
import os.path as osp
import warnings
import math

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

# test
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmpose.apis import multi_gpu_test, single_gpu_test

import custom_tools.supervise_tool as supt
import tools.custom_tools.old.test_tool_old as tst
from custom_tools.base_tool import create_custom_file, create_dir
from custom_tools.base_tool import merge_configs
from custom_tools.test_tools import create_test_qualitative_images, create_test_quantitative_results
from custom_tools.hyperparameter_search_engine import HyperparameterSearchEngine
from custom_tools.base_tool import save_lr_distribution
from pixelconversor.conversor.searcher.utils import base as bs
from pixelconversor.conversor.searcher.searchers import modelSearcher

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
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
    args.config = "../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_23kp_256x192.py"
    args.cfg_options = {"model.pretrained": "C:/Users/Tecnico/Downloads/vitpose-h.pth"}
    # args.work_dir = "D:/vitpose_work_dir/exp_results/_23KP_EXPS/LATERAL/PRUEBA"
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


args = parse_args()
cfg = Config.fromfile(args.config)


def prepare_environment():
    dt_name = os.path.basename(cfg.data_root)
    splits = dt_name.split("_")
    res_dir = cfg.results_dir
    if len(splits) == 9:
        name = "experiment_" + splits[4] + "_" + splits[5] + "_" + splits[7] + "_" + splits[8]
        sub_dir = res_dir + "/" + splits[3] + "/" + splits[2] + "/" + name
        args.work_dir = sub_dir
        create_dir(sub_dir)
        create_dir(os.path.join(sub_dir, "hyperparameter_search"))
    print("IMASHRIMP: Preparing environment--------------------------------")
    print("\tWork dir: ", args.work_dir)


def hyperparameter_search():
    print("IMASHRIMP: Searching hyperparameters-----------------------------")
    # bch, lr, total_epochs = HyperparameterSearchEngine(args, cfg)()
    bch = 16
    lr = 0.0002
    total_epochs = 486
    cfg.optimizer['lr'] = lr
    cfg.data['samples_per_gpu'] = bch
    cfg.data['val_dataloader']['samples_per_gpu'] = bch
    cfg.data['test_dataloader']['samples_per_gpu'] = bch
    cfg.total_epochs = total_epochs
    in_1 = math.ceil(((170 / 2.1) * cfg.total_epochs) / 100)
    in_2 = math.ceil(((200 / 2.1) * cfg.total_epochs) / 100)
    cfg.lr_config['step'] = [in_1, in_2]
    source_dir = args.work_dir + ""

    args.work_dir = os.path.join(source_dir, "winner_" + str(bch) + "_" + str(lr) + "_" + str(cfg.total_epochs))
    resume_from_file = ""
    if os.path.exists(args.work_dir):
        resume_from_dir = os.path.join(source_dir, "winner_" + str(bch) + "_" + str(lr) + "_" + str(cfg.total_epochs))
        dir_list = os.listdir(resume_from_dir)
        for file in dir_list:
            if "best_LOSS" in file:
                resume_from_file = os.path.join(resume_from_dir, file)
                break

    if not os.path.exists(args.work_dir) or resume_from_file == "":
        resume_from_dir = source_dir + "/hyperparameter_search/exp_" + str(bch) + "_" + str(lr) + "_" + str(
            cfg.total_epochs) + "_" + str(5)
        dir_list = os.listdir(resume_from_dir)
        for file in dir_list:
            if "best_LOSS" in file:
                resume_from_file = os.path.join(resume_from_dir, file)
                break

    if not resume_from_file == "":
        args.resume_from = resume_from_file

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    print("\tWinner batch size: ", bch)
    print("\tWinner learning rate: ", lr)
    print("\tWinner total epochs: ", total_epochs)


def train():
    print("IMASHRIMP: Training--------------------------------")
    inicio = time.time()

    if os.path.exists(args.work_dir):
        fl = os.listdir(args.work_dir)
        for f in fl:
            if "4_Train_Time" in f:
                return

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )
    train_lrs = train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=False,
        timestamp=timestamp,
        meta=meta)
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    horas = int(tiempo_transcurrido // 3600)
    minutos = int((tiempo_transcurrido % 3600) // 60)
    create_custom_file(os.path.join(cfg.work_dir, "4_Train_Time_" + str(horas) + "h_" + str(minutos) + "m.txt"), "")
    print(f"\tTiempo de entrenamiento: {horas} horas y {minutos} minutos")
    file_name = os.path.join(args.work_dir, 'learning_rates.json')
    with open(file_name, 'w') as f:
        json.dump(train_lrs, f)
    # save_lr_distribution(train_lrs, path_to_save=args.work_dir)


def supervise():
    if args.work_dir:
        list_f = os.listdir(args.work_dir)
        for f in list_f:
            if "supervise_info" in f:
                supervise_info_path = os.path.join(args.work_dir, f)
    else:
        print("supervise_info info not exist")
        pass
    with open(supervise_info_path, 'r') as f:
        datos = json.load(f)

    t_loss = datos["t_loss"]
    e_loss = datos["e_loss"]
    supt.save_supervise_loss(t_loss, e_loss, "LOSS", args.work_dir, thr=1)

    # pt_loss = datos["pt_loss"]
    # pe_loss = datos["pe_loss"]
    # supt.save_supervise_loss(pt_loss, pe_loss, "LOSS WIN_20", args.work_dir)

    t_pck = datos["t_pck"]
    e_pck = datos["e_pck"]
    supt.save_supervise_pck(t_pck, e_pck, "PCK", args.work_dir)


def preapare_conversion_model():
    print("IMASHRIMP: Preparing conversion model--------------------------------")
    # Fallo: Si se hace primero la conversión con un dataset de 22kp. La conversión 23kp falla.
    model_dir = bs.get_model_dir(cfg)
    if not os.path.exists(model_dir):
        results_dic = bs.get_ground_truth_measures(cfg)
        modelSearcher.PixelToCentimeterModelSearcher(cfg).search_best_model(results_dic)
    cfg.model_add = model_dir + "/model_results.json"
    print("\tModel dir: ", model_dir)


def test(complete=False):
    print(f"IMASHRIMP: Testing {'' if not complete else 'complete dataset'}--------------------------------")
    args.cfg_options = {}
    args.launcher = 'none'
    if args.work_dir:
        list_f = os.listdir(args.work_dir)
        checkpoints = []
        yet_best = []
        for f in list_f:
            if "best" in f and ".pth" in f:
                splits = f.split("_")
                ep = splits[3][:-4]
                if ep not in yet_best:
                    checkpoints.append(os.path.join(args.work_dir, f))
                    yet_best.append(ep)
    cfg_data = cfg.data.test
    if complete:
        cfg_data = cfg.data.total

    for checkpoint in checkpoints:
        checkpoint_name = os.path.basename(checkpoint)[
                          :-4] if not complete else f"{os.path.basename(checkpoint)[:-4]}_complete"
        final_file = os.path.join(cfg.work_dir, "_test_quantitative_" + checkpoint_name + ".txt")
        if os.path.exists(final_file):
            continue
        args.checkpoint = checkpoint
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
        cfg_data.test_mode = True

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
        dataset = build_dataset(cfg_data, dict(test_mode=True))
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
            inicio = time.time()
            outputs = single_gpu_test(model, data_loader)
            fin = time.time()
            tiempo_ejecucion = fin - inicio
            # print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            inicio = time.time()
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

            fin = time.time()

            tiempo_ejecucion = fin - inicio
            # print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")

        rank, _ = get_dist_info()
        eval_config = cfg.get('evaluation', {})
        eval_config = merge_configs(eval_config, dict(metric=args.eval))

        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            # tst.get_measure_info(cfg.ann_file_measure, outputs)
            results = dataset.evaluate(outputs, cfg.work_dir, err_dis=True, **eval_config)

            pd_mae, gt_mae = create_test_quantitative_results(outputs, dataset, checkpoint_name, cfg)
            # if args.predict_images:
            #     create_test_qualitative_images(outputs, dataset, checkpoint_name, cfg)
            # tst.save_error_per_point_histogram(results['PCKdis'], os.path.join(args.work_dir, "1_ERR_DIS_NEW_" + checkpoint_name + ".png"), metric="px")
            del results['PCKdis']
            results['mae_pd_rm'] = pd_mae
            results['mae_gt_rm'] = gt_mae
            create_custom_file(os.path.join(cfg.work_dir, "_test_quantitative_" + checkpoint_name + ".txt"), results)


if __name__ == '__main__':
    prepare_environment()
    hyperparameter_search()
    train()
    preapare_conversion_model()
    supervise()
    test()
    # if cfg.complete_analysis:
    #     test(complete=True)
