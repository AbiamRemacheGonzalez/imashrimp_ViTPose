# Copyright (c) OpenMMLab. All rights reserved.
from pylab import *
import json
import copy
import time
from imashrimp_mmcv.mmcv.runner import set_random_seed
from imashrimp_mmcv.mmcv.utils import get_git_hash

from imashrimp_ViTPose.mmpose import __version__
# from imashrimp_ViTPose.mmpose.apis import init_random_seed, train_model
from imashrimp_ViTPose.mmpose.apis import init_random_seed_for_search, train_model_for_search
from imashrimp_ViTPose.mmpose.utils import collect_env
import os
import os.path as osp
import warnings

import imashrimp_mmcv.mmcv as mmcv
import torch
from imashrimp_mmcv.mmcv import Config
from imashrimp_mmcv.mmcv.runner import get_dist_info, init_dist

from imashrimp_ViTPose.mmpose.datasets import build_dataset
from imashrimp_ViTPose.mmpose.models import build_posenet
from imashrimp_ViTPose.mmpose.utils import setup_multi_processes

# test
from tools.custom_tools.base_tool import create_custom_file

#
# supervise
import pandas as pd
import math

#
try:
    from imashrimp_mmcv.mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from imashrimp_ViTPose.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from imashrimp_ViTPose.mmpose.core import wrap_fp16_model

warnings.filterwarnings("ignore", category=UserWarning)


class HyperparameterSearchEngine:
    def __init__(self, args, cfg):
        self.args = copy.deepcopy(args)
        self.cfg = Config(copy.deepcopy(cfg))
        self.iters = 22890  #77280#22890 Para redes mas grandes. De momento este será el valor.
        self.iters_for_bigs = 72280#77280
        datasets = [build_dataset(self.cfg.data.train)]
        self.train_len = len(datasets[0])

    def train_n_ep(self, bch, lr, nep):
        inicio = time.time()
        args = self.args

        cfg = self.cfg
        cfg.optimizer['lr'] = lr
        cfg.data['samples_per_gpu'] = bch
        cfg.data['val_dataloader']['samples_per_gpu'] = bch
        cfg.data['test_dataloader']['samples_per_gpu'] = bch

        # set multi-process settings
        setup_multi_processes(cfg)

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

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
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        meta = dict()
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        meta['env_info'] = env_info

        # logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        seed = init_random_seed_for_search(args.seed)
        set_random_seed(seed, deterministic=args.deterministic)
        cfg.seed = seed
        meta['seed'] = seed

        model = build_posenet(cfg.model)
        datasets = [build_dataset(cfg.data.train)]
        # ----------
        cfg.total_epochs = math.ceil((self.iters * bch) / len(datasets[0]))
        if cfg.total_epochs < 210:
            self.iters = self.iters_for_bigs
            cfg.total_epochs = math.ceil((self.iters * bch) / len(datasets[0]))
        in_1 = math.ceil(((170/2.1)*cfg.total_epochs)/100)
        in_2 = math.ceil(((200/2.1)*cfg.total_epochs)/100)
        cfg.lr_config['step'] = [in_1, in_2]
        iters_per_exp = math.ceil(len(datasets[0]) / self.min_bch) * nep
        exp_epochs = math.ceil(iters_per_exp / math.ceil((len(datasets[0]) / bch)))

        args.work_dir = os.path.join(args.work_dir, "exp_" + str(bch) + "_" + str(lr) + "_" + str(cfg.total_epochs) + "_" + str(nep))

        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        # -----------

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
        train_lrs = train_model_for_search(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=False,
            timestamp=timestamp,
            meta=meta,
            exp_epochs=exp_epochs)

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

        fin = time.time()
        tiempo_transcurrido = fin - inicio
        horas = int(tiempo_transcurrido // 3600)
        minutos = int((tiempo_transcurrido % 3600) // 60)
        create_custom_file(os.path.join(cfg.work_dir, "4_Train_Time_" + str(horas) + "h_" + str(minutos) + "m.txt"), "")
        tiempo = str(horas) + "h_" + str(minutos) + "m"
        file_name = os.path.join(self.args.work_dir, 'learning_rates.json')
        with open(file_name, 'w') as f:
            json.dump(train_lrs, f)

    def already_exists(self, bch, lr, nep):
        exp_list = os.listdir(self.source_dir)
        for exp in exp_list:
            split = exp.split("_")
            if int(split[1]) == bch and float(split[2]) == lr and int(split[4]) == nep:
                total_epochs = math.ceil((self.iters * bch) / self.train_len)
                if total_epochs < 210:
                    self.iters = self.iters_for_bigs
                    total_epochs = math.ceil((self.iters * bch) / self.train_len)
                dir = os.path.join(self.source_dir, "exp_" + str(bch) + "_" + str(lr) + "_" + str(total_epochs) + "_" + str(nep))
                if os.path.exists(dir):
                    list_dir = os.listdir(dir)
                    for file in list_dir:
                        if "4_Train_Time" in file:
                            return True
        return False

    def get_exp_info(self, bch, lr, nep):
        total_epochs = math.ceil((self.iters * bch) / self.train_len)
        if total_epochs < 210:
            self.iters = self.iters_for_bigs
            total_epochs = math.ceil((self.iters * bch) / self.train_len)
        work_dir = os.path.join(self.source_dir, "exp_" + str(bch) + "_" + str(lr) + "_" + str(total_epochs) + "_" + str(nep))
        if work_dir:
            list_f = os.listdir(work_dir)
            for f in list_f:
                if "supervise_info" in f:
                    supervise_info_path = os.path.join(work_dir, f)
        else:
            print("supervise_info info not exist")
            pass
        with open(supervise_info_path, 'r') as f:
            datos = json.load(f)
        return datos["e_loss"], datos["e_pck"]

    @staticmethod
    def delete_pth_files(ruta_inicial, carpeta_excepcion=None):
        carpeta_excepcion = os.path.abspath(carpeta_excepcion) if carpeta_excepcion else None
        # Recorre la ruta dada, incluyendo subdirectorios
        for carpeta_raiz, subcarpetas, ficheros in os.walk(ruta_inicial):
            if carpeta_excepcion and os.path.abspath(carpeta_raiz) == carpeta_excepcion:
                continue
            for fichero in ficheros:
                # Verifica si el archivo termina en '.pth'
                if fichero.endswith('.pth'):
                    # Obtiene la ruta completa del archivo
                    ruta_completa = os.path.join(carpeta_raiz, fichero)
                    try:
                        # Elimina el archivo
                        os.remove(ruta_completa)
                        print(f"Se pudo eliminar el archivo {ruta_completa}")
                    except Exception as e:
                        print(f"No se pudo eliminar el archivo {ruta_completa}: {e}")

    def __call__(self):
        copy_args = copy.deepcopy(self.args)
        copy_cfg = Config(copy.deepcopy(self.cfg))

        self.source_dir = self.args.work_dir + "/hyperparameter_search"
        self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
        lrs = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
        bchs = [16]#[8, 16] # 20 23 4 no parecen una buena opción
        self.min_bch = 8
        res_1ep = []

        first_batch = 1
        for bch in bchs:
            for lr in lrs:
                if not self.already_exists(bch, lr, first_batch):
                    self.train_n_ep(bch, lr, first_batch)
                    self.delete_pth_files(self.args.work_dir)
                    self.args = copy.deepcopy(copy_args)
                    self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
                    self.cfg = Config(copy.deepcopy(copy_cfg))
                e_loss, e_pck = self.get_exp_info(bch, lr, first_batch)
                idx = e_loss.index(min(e_loss))
                res_1ep.append({
                    "config": {"bch": bch, "lr": lr},
                    "e_loss": min(e_loss),
                    "e_pck": e_pck[idx]
                })
        df_res_1ep = pd.DataFrame(res_1ep)
        sel = math.ceil(len(res_1ep)/4)
        top_20_loss = df_res_1ep.nsmallest(sel, 'e_loss')
        win_configs = top_20_loss['config'].to_list()
        res_5ep = []
        second_batch = 5
        for config in win_configs:
            bch = config['bch']
            lr = config['lr']
            if not self.already_exists(bch, lr, second_batch):
                self.train_n_ep(bch, lr, second_batch)
                # self.delete_pth_files(self.args.work_dir)
                self.args = copy.deepcopy(copy_args)
                self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
                self.cfg = Config(copy.deepcopy(copy_cfg))

            e_loss, e_pck = self.get_exp_info(bch, lr, second_batch)
            idx = e_loss.index(min(e_loss))
            res_5ep.append({
                "config": {"bch": bch, "lr": lr},
                "e_loss": min(e_loss),
                "e_pck": e_pck[idx]
            })
        df_res_5ep = pd.DataFrame(res_5ep)
        top_1_loss = df_res_5ep.nsmallest(1, 'e_loss')
        win_config = top_1_loss['config'].to_list()

        bch_win = win_config[0]['bch']
        lr_win = win_config[0]['lr']
        total_epochs_win = math.ceil((self.iters * win_config[0]['bch']) / self.train_len)

        self.args = copy.deepcopy(copy_args)
        self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
        self.cfg = Config(copy.deepcopy(copy_cfg))

        win_dir = os.path.join(self.args.work_dir, "exp_" + str(bch_win) + "_" + str(lr_win) + "_" + str(total_epochs_win) + "_" + str(5))
        self.delete_pth_files(self.args.work_dir, carpeta_excepcion=win_dir)
        return bch_win, lr_win, total_epochs_win
