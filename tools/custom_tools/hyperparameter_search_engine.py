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
from decimal import Decimal

# test
from .base_tool import create_custom_file

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
        datasets = [build_dataset(self.cfg.data.train)]
        self.train_len = len(datasets[0])
        factor = 2
        self.total_epochs_win = 210 * factor
        self.lr_config = [170 * factor, 200 * factor]
        self.num_batchs = 1

    def train_n_ep(self, bch, lr, nep, bch_search=False):
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

        # set random seeds
        seed = init_random_seed_for_search(args.seed)
        set_random_seed(seed, deterministic=args.deterministic)
        cfg.seed = seed
        meta['seed'] = seed

        model = build_posenet(cfg.model)
        datasets = [build_dataset(cfg.data.train)]

        cfg.total_epochs = self.total_epochs_win
        cfg.lr_config['step'] = self.lr_config
        if bch_search:
            args.work_dir = os.path.join(args.work_dir,
                                         "bch_" + str(bch) + "_" + str(lr) + "_" + str(cfg.total_epochs) + "_" + str(
                                             nep))
        else:
            args.work_dir = os.path.join(args.work_dir,
                                         "exp_" + str(bch) + "_" + str(lr) + "_" + str(cfg.total_epochs) + "_" + str(
                                             nep))
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir

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
        if bch_search:
            cfg.workflow = [('bch_search', 1)]
            datasets = [build_dataset(cfg.data.train)]

        train_lrs = train_model_for_search(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=False,
            timestamp=timestamp,
            meta=meta,
            exp_epochs=nep)

        fin = time.time()
        tiempo_transcurrido = fin - inicio
        horas = int(tiempo_transcurrido // 3600)
        minutos = int((tiempo_transcurrido % 3600) // 60)
        create_custom_file(os.path.join(cfg.work_dir, "4_Train_Time_" + str(horas) + "h_" + str(minutos) + "m.txt"), "")
        file_name = os.path.join(self.args.work_dir, 'learning_rates.json')
        with open(file_name, 'w') as f:
            json.dump(train_lrs, f)

    def already_exists(self, bch, lr, nep, bch_search=False):
        exp_list = os.listdir(self.source_dir)
        if "memory_usage.csv" in exp_list:
            exp_list.remove("memory_usage.csv")
        prefix = "bch" if bch_search else "exp"
        for exp in exp_list:
            split = exp.split("_")
            if split[0] == prefix and int(split[1]) == bch and float(split[2]) == lr and int(split[4]) == nep:
                dir = os.path.join(self.source_dir, exp)
                if os.path.exists(dir):
                    list_dir = os.listdir(dir)
                    for file in list_dir:
                        if "4_Train_Time" in file:
                            return True
        return False

    def get_exp_info(self, bch, lr, nep):
        exp_list = os.listdir(self.source_dir)
        supervise_info_path = None
        for exp in exp_list:
            split = exp.split("_")
            if split[0] == "exp" and int(split[1]) == bch and float(split[2]) == lr and int(split[4]) == nep:
                dir = os.path.join(self.source_dir, exp)
                if os.path.exists(dir):
                    list_dir = os.listdir(dir)
                    for f in list_dir:
                        if "supervise_info" in f:
                            supervise_info_path = os.path.join(dir, f)
                            break
                break
        with open(supervise_info_path, 'r') as f:
            datos = json.load(f)
        return datos["e_loss"], datos["e_pck"]

    def get_memory_usage(self, bch, lr, nep):
        exp_list = os.listdir(self.source_dir)
        if "memory_usage.csv" in exp_list:
            exp_list.remove("memory_usage.csv")
        log_files = []
        for exp in exp_list:
            split = exp.split("_")
            if int(split[1]) == bch and float(split[2]) == lr and int(split[4]) == nep:
                dir = os.path.join(self.source_dir, exp)
                if os.path.exists(dir):
                    list_dir = os.listdir(dir)
                    for f in list_dir:
                        if ".log" in f:
                            supervise_info_path = os.path.join(dir, f)
                            log_files.append(supervise_info_path)
                # break
        for log_file in log_files:
            env_info, data = self.read_log_file(log_file)
            if data != []:
                break
        last_memory = data[-1]['memory']
        return last_memory

    def read_log_file(self, path):
        data = []
        env_info = {}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                # Ignoramos líneas vacías si las hubiera
                if not line.strip():
                    continue

                try:
                    # Parseamos la línea actual
                    item = json.loads(line)

                    # Como tu primera línea es configuración y el resto son métricas,
                    # tal vez quieras separarlas:
                    if i == 0 and "env_info" in item:
                        env_info = item
                    else:
                        data.append(item)

                except json.JSONDecodeError as e:
                    print(f"Error en la línea {i + 1}: {e}")
        return env_info, data

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

    def winner_already_exists(self):
        path = self.args.work_dir
        dirs = os.listdir(path)
        for dir in dirs:
            if "winner_" in dir:
                winner_path = os.path.join(path, dir)
                list_dir = os.listdir(winner_path)
                for file in list_dir:
                    if "4_Train_Time_" in file:
                        return True
        return False

    def get_winner_params(self):
        dirs = os.listdir(self.args.work_dir)
        for dir in dirs:
            if "winner_" in dir:
                split = dir.split("_")
                bch_win = int(split[1])
                lr_win = float(split[2])
                total_epochs_win = int(split[3])
                return bch_win, lr_win, total_epochs_win

    def __call__(self):
        copy_args = copy.deepcopy(self.args)
        copy_cfg = Config(copy.deepcopy(self.cfg))
        if self.winner_already_exists():
            bch_win, lr_win, total_epochs_win = self.get_winner_params()
            return bch_win, lr_win, total_epochs_win
        self.source_dir = self.args.work_dir + "/hyperparameter_search"
        self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
        # lrs = [0.00001, 0.00003, 0.00005, 0.00007, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01] # For Huge models
        lrs = [0.00001, 0.00003, 0.00005, 0.00007, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01,
               0.05, 0.07, 0.1, 0.5, 0.7, 1]  # For Small models

        # Find optimal batch size
        tests = [8, 16, 32, 64, 128]  # 128 , 256, 512, 1024
        memory_results = []

        slr = 0.0001
        best_bch = 8
        max_memory = 24 * 1024  # in MB
        for bch in tests:
            if not self.already_exists(bch, slr, self.num_batchs, bch_search=True):
                self.train_n_ep(bch, slr, self.num_batchs, bch_search=True)
                self.delete_pth_files(self.args.work_dir)
                self.args = copy.deepcopy(copy_args)
                self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
                self.cfg = Config(copy.deepcopy(copy_cfg))
            memory_used = self.get_memory_usage(bch, slr, self.num_batchs)
            memory_results.append({
                "bch": bch,
                "memory_used": memory_used
            })
            if memory_used < max_memory:
                best_bch = bch
                if memory_used > max_memory * 0.9:
                    break
            if memory_used >= max_memory:
                break
        df_memory = pd.DataFrame(memory_results)
        df_memory.to_csv(os.path.join(self.source_dir, "memory_usage.csv"), index=False)

        # Get optimal learning rate
        res_1ep = []
        for lr in lrs:
            if not self.already_exists(best_bch, lr, self.num_batchs):
                self.train_n_ep(best_bch, lr, self.num_batchs)
                self.delete_pth_files(self.args.work_dir)
                self.args = copy.deepcopy(copy_args)
                self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
                self.cfg = Config(copy.deepcopy(copy_cfg))
            e_loss, e_pck = self.get_exp_info(best_bch, lr, self.num_batchs)
            idx = e_loss.index(min(e_loss))
            res_1ep.append({
                "bch": best_bch,
                "lr": lr,
                "e_loss": min(e_loss),
                "e_pck": e_pck[idx]
            })

        df_res_1ep = pd.DataFrame(res_1ep)
        top_n = df_res_1ep.sort_values(['e_pck', 'e_loss'], ascending=[False, True]).head(4)
        selected_lrs = [Decimal(str(x)) / Decimal("10") for x in top_n['lr'].to_list()]
        selected_lrs = [float(x) for x in selected_lrs]
        while True:
            df_res_1ep = pd.DataFrame(res_1ep)
            all_lrs = df_res_1ep['lr'].to_list()
            not_experimented_lrs = list(set(selected_lrs) - set(all_lrs))
            if len(not_experimented_lrs) == 0:
                break
            for lr in not_experimented_lrs:
                if not self.already_exists(best_bch, lr, self.num_batchs):
                    self.train_n_ep(best_bch, lr, self.num_batchs)
                    self.delete_pth_files(self.args.work_dir)
                    self.args = copy.deepcopy(copy_args)
                    self.args.work_dir = self.args.work_dir + "/hyperparameter_search"
                    self.cfg = Config(copy.deepcopy(copy_cfg))
                e_loss, e_pck = self.get_exp_info(best_bch, lr, self.num_batchs)
                idx = e_loss.index(min(e_loss))
                res_1ep.append({
                    "bch": best_bch,
                    "lr": lr,
                    "e_loss": min(e_loss),
                    "e_pck": e_pck[idx]
                })

        selected_exps = df_res_1ep[df_res_1ep['lr'].isin(selected_lrs)]
        top_1 = selected_exps.sort_values(['e_pck', 'e_loss'], ascending=[False, True]).head(1)
        lr_win = top_1['lr'].values[0]
        bch_win = top_1['bch'].values[0]
        return bch_win, lr_win, self.total_epochs_win
