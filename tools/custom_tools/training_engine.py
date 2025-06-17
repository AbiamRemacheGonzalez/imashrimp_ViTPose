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

# import supervise_tool as supt
from .supervise_tool import save_supervise_loss, save_supervise_pck
from .base_tool import create_custom_file, create_dir
from .base_tool import merge_configs
from .test_tools import create_test_qualitative_images, create_test_quantitative_results
from .hyperparameter_search_engine import HyperparameterSearchEngine
from pixelconversor.conversor.searcher.utils import base as bs
from pixelconversor.conversor.searcher.searchers import modelSearcher

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

warnings.filterwarnings("ignore", category=UserWarning)


class TrainingEngine:
    def __init__(self, args, cfg, complete_system=False):
        self.args = copy.deepcopy(args)
        self.cfg = Config(copy.deepcopy(cfg))
        self.cs = complete_system

    def __call__(self):
        self.prepare_environment(complete_system=self.cs)
        self.hyperparameter_search()
        self.train()
        self.supervise()
        self.preapare_conversion_model()
        self.test()
        # if self.cfg.complete_analysis:
        #     self.test(complete=True)

    def external_test(self, source_dir, res_dir, view, nkp, old=False):
        self.prepare_environment(old=old)
        self.preapare_conversion_model()
        listdir = os.listdir(self.args.work_dir)
        for dir in listdir:
            if "winner" in dir:
                self.args.work_dir = os.path.join(self.args.work_dir, dir)
        data_root = source_dir + f"/{view}_{nkp}"
        res_dir = res_dir + f"/{view}_{nkp}"
        self.cfg.data_root = data_root
        self.cfg.data.test.ann_file = f'{data_root}/annotations/test_keypoints.json'
        self.cfg.data.test.img_prefix = f'{data_root}/images/test/'
        self.cfg.data.test.img_prefix_depth = f'{data_root}/depths/test/'

        self.cfg.work_dir = res_dir
        data_dir = self.test(external=True)
        return data_dir

    def prepare_environment(self, complete_system=False, old=False):
        dt_name = os.path.basename(self.cfg.data_root)
        splits = dt_name.split("_")
        res_dir = self.cfg.results_dir
        if len(splits) == 9:
            name = "experiment_" + splits[4] + "_" + splits[5] + "_" + splits[7] + "_" + splits[8]
            if old:
                name = "old_experiment_" + splits[4] + "_" + splits[5] + "_" + splits[7] + "_" + splits[8]
            if complete_system:
                sub_dir = res_dir + f"/{splits[3]}_{splits[2]}/{name}"
            else:
                sub_dir = res_dir + "/" + splits[3] + "/" + splits[2] + "/" + name
            self.args.work_dir = sub_dir
            self.cfg.work_dir = sub_dir
            create_dir(sub_dir)
            create_dir(os.path.join(sub_dir, "hyperparameter_search"))
        print("IMASHRIMP: Preparing environment--------------------------------")
        print("\tWork dir: ", self.cfg.work_dir)

    def hyperparameter_search(self):
        print("IMASHRIMP: Searching hyperparameters-----------------------------")
        bch, lr, total_epochs = HyperparameterSearchEngine(self.args, self.cfg)()
        self.cfg.optimizer['lr'] = lr
        self.cfg.data['samples_per_gpu'] = bch
        self.cfg.data['val_dataloader']['samples_per_gpu'] = bch
        self.cfg.data['test_dataloader']['samples_per_gpu'] = bch
        self.cfg.total_epochs = total_epochs
        in_1 = math.ceil(((170 / 2.1) * self.cfg.total_epochs) / 100)
        in_2 = math.ceil(((200 / 2.1) * self.cfg.total_epochs) / 100)
        self.cfg.lr_config['step'] = [in_1, in_2]
        source_dir = self.args.work_dir + ""

        self.args.work_dir = os.path.join(source_dir,
                                          "winner_" + str(bch) + "_" + str(lr) + "_" + str(self.cfg.total_epochs))
        resume_from_file = ""
        if os.path.exists(self.args.work_dir):
            resume_from_dir = os.path.join(source_dir,
                                           "winner_" + str(bch) + "_" + str(lr) + "_" + str(self.cfg.total_epochs))
            dir_list = os.listdir(resume_from_dir)
            for file in dir_list:
                if "best_PCK" in file:
                    resume_from_file = os.path.join(resume_from_dir, file)
                    break

        if not os.path.exists(self.args.work_dir) or resume_from_file == "":
            resume_from_dir = source_dir + "/hyperparameter_search/exp_" + str(bch) + "_" + str(lr) + "_" + str(
                self.cfg.total_epochs) + "_" + str(5)
            dir_list = os.listdir(resume_from_dir)
            for file in dir_list:
                if "best_PCK" in file:
                    resume_from_file = os.path.join(resume_from_dir, file)
                    break

        if not resume_from_file == "":
            self.args.resume_from = resume_from_file

        if self.args.work_dir is not None:
            self.cfg.work_dir = self.args.work_dir
        print("\tWinner batch size: ", bch)
        print("\tWinner learning rate: ", lr)
        print("\tWinner total epochs: ", total_epochs)

    def train(self):
        print("IMASHRIMP: Training--------------------------------")
        inicio = time.time()

        if os.path.exists(self.args.work_dir):
            fl = os.listdir(self.args.work_dir)
            for f in fl:
                if "4_Train_Time" in f:
                    return

        if self.args.cfg_options is not None:
            self.cfg.merge_from_dict(self.args.cfg_options)

        # set multi-process settings
        setup_multi_processes(self.cfg)

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined in this priority: CLI > segment in file > filename
        if self.args.work_dir is not None:
            # update configs according to CLI self.args if self.args.work_dir is not None
            self.cfg.work_dir = self.args.work_dir
        elif self.cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if self.cfg.work_dir is None
            self.cfg.work_dir = osp.join('./work_dirs',
                                         osp.splitext(osp.basename(self.args.config))[0])
        if self.args.resume_from is not None:
            self.cfg.resume_from = self.args.resume_from
        if self.args.gpus is not None:
            self.cfg.gpu_ids = range(1)
            warnings.warn('`--gpus` is deprecated because we only support '
                          'single GPU mode in non-distributed training. '
                          'Use `gpus=1` now.')
        if self.args.gpu_ids is not None:
            self.cfg.gpu_ids = self.args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed training. Use the first GPU '
                          'in `gpu_ids` now.')
        if self.args.gpus is None and self.args.gpu_ids is None:
            self.cfg.gpu_ids = [self.args.gpu_id]

        if self.args.autoscale_lr:
            # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
            self.cfg.optimizer['lr'] = self.cfg.optimizer['lr'] * len(self.cfg.gpu_ids) / 8

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher == 'none':
            distributed = False
            if len(self.cfg.gpu_ids) > 1:
                warnings.warn(
                    f'We treat {self.cfg.gpu_ids} as gpu-ids, and reset to '
                    f'{self.cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                    'non-distribute training time.')
                self.cfg.gpu_ids = self.cfg.gpu_ids[0:1]
        else:
            distributed = True
            init_dist(self.args.launcher, **self.cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

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
        logger.info(f'Config:\n{self.cfg.pretty_text}')

        # set random seeds
        seed = init_random_seed(self.args.seed)
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {self.args.deterministic}')
        set_random_seed(seed, deterministic=self.args.deterministic)
        self.cfg.seed = seed
        meta['seed'] = seed

        model = build_posenet(self.cfg.model)
        datasets = [build_dataset(self.cfg.data.train)]

        if len(self.cfg.workflow) == 2:
            val_dataset = copy.deepcopy(self.cfg.data.val)
            val_dataset.pipeline = self.cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))

        if self.cfg.checkpoint_config is not None:
            # save mmpose version, config file content
            # checkpoints as meta data
            self.cfg.checkpoint_config.meta = dict(
                mmpose_version=__version__ + get_git_hash(digits=7),
                config=self.cfg.pretty_text,
            )
        train_lrs = train_model(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=False,
            timestamp=timestamp,
            meta=meta)
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        horas = int(tiempo_transcurrido // 3600)
        minutos = int((tiempo_transcurrido % 3600) // 60)
        create_custom_file(
            os.path.join(self.cfg.work_dir, "4_Train_Time_" + str(horas) + "h_" + str(minutos) + "m.txt"), "")
        print(f"\tTiempo de entrenamiento: {horas} horas y {minutos} minutos")
        file_name = os.path.join(self.args.work_dir, 'learning_rates.json')
        with open(file_name, 'w') as f:
            json.dump(train_lrs, f)
        # save_lr_distribution(train_lrs, path_to_save=self.args.work_dir)

    def supervise(self):
        if self.args.work_dir:
            list_f = os.listdir(self.args.work_dir)
            for f in list_f:
                if "supervise_info" in f:
                    supervise_info_path = os.path.join(self.args.work_dir, f)
        else:
            print("supervise_info info not exist")
            pass
        with open(supervise_info_path, 'r') as f:
            datos = json.load(f)

        t_loss = datos["t_loss"]
        e_loss = datos["e_loss"]
        save_supervise_loss(t_loss, e_loss, "LOSS", self.args.work_dir, thr=1)

        # pt_loss = datos["pt_loss"]
        # pe_loss = datos["pe_loss"]
        # supt.save_supervise_loss(pt_loss, pe_loss, "LOSS WIN_20", self.args.work_dir)

        t_pck = datos["t_pck"]
        e_pck = datos["e_pck"]
        save_supervise_pck(t_pck, e_pck, "PCK", self.args.work_dir)

    def preapare_conversion_model(self):
        print("IMASHRIMP: Preparing conversion model--------------------------------")
        # Fallo: Si se hace primero la conversión con un dataset de 22kp. La conversión 23kp falla.
        model_dir = bs.get_model_dir(self.cfg)
        if not os.path.exists(model_dir):
            results_dic = bs.get_ground_truth_measures(self.cfg)
            modelSearcher.PixelToCentimeterModelSearcher(self.cfg).search_best_model(results_dic)
        self.cfg.model_add = model_dir + "/model_results.json"
        print("\tModel dir: ", model_dir)

    def test(self, complete=False, external=False):
        print(f"IMASHRIMP: Testing {'' if not complete else 'complete dataset'}--------------------------------")
        self.args.cfg_options = {}
        self.args.launcher = 'none'
        if self.args.work_dir:
            list_f = os.listdir(self.args.work_dir)
            checkpoints = []
            yet_best = []
            for f in list_f:
                if "best" in f and ".pth" in f:
                    splits = f.split("_")
                    ep = splits[3][:-4]
                    if ep not in yet_best:
                        checkpoints.append(os.path.join(self.args.work_dir, f))
                        yet_best.append(ep)
        cfg_data = self.cfg.data.test
        if complete:
            cfg_data = self.cfg.data.total

        for checkpoint in checkpoints:
            checkpoint_name = os.path.basename(checkpoint)[
                              :-4] if not complete else f"{os.path.basename(checkpoint)[:-4]}_complete"
            final_file = os.path.join(self.cfg.work_dir, "_test_quantitative_" + checkpoint_name + ".txt")
            if os.path.exists(final_file):
                continue
            self.args.checkpoint = checkpoint
            self.args.fuse_conv_bn = None
            self.args.gpu_id = 0
            self.args.tmpdir = None
            self.args.gpu_collect = True
            self.args.eval = None
            self.args.out = None

            if self.args.cfg_options is not None:
                self.cfg.merge_from_dict(self.args.cfg_options)

            # set multi-process settings
            setup_multi_processes(self.cfg)

            # set cudnn_benchmark
            if self.cfg.get('cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = True
            self.cfg.model.pretrained = None
            cfg_data.test_mode = True

            # work_dir is determined in this priority: CLI > segment in file > filename
            if self.args.work_dir is not None and not external:
                # update configs according to CLI self.args if self.args.work_dir is not None
                self.cfg.work_dir = self.args.work_dir
            elif self.cfg.get('work_dir', None) is None:
                # use config filename as default work_dir if self.cfg.work_dir is None
                self.cfg.work_dir = osp.join('./work_dirs',
                                             osp.splitext(osp.basename(self.args.config))[0])

            mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))

            # init distributed env first, since logger depends on the dist info.
            if self.args.launcher == 'none':
                distributed = False
            else:
                distributed = True
                init_dist(self.args.launcher, **self.cfg.dist_params)

            # build the dataloader
            dataset = build_dataset(cfg_data, dict(test_mode=True))
            # step 1: give default values and override (if exist) from self.cfg.data
            loader_cfg = {
                **dict(seed=self.cfg.get('seed'), drop_last=False, dist=distributed),
                **({} if torch.__version__ != 'parrots' else dict(
                    prefetch_num=2,
                    pin_memory=False,
                )),
                **dict((k, self.cfg.data[k]) for k in [
                    'seed',
                    'prefetch_num',
                    'pin_memory',
                    'persistent_workers',
                ] if k in self.cfg.data)
            }
            # step2: self.cfg.data.test_dataloader has higher priority
            test_loader_cfg = {
                **loader_cfg,
                **dict(shuffle=False, drop_last=False),
                **dict(workers_per_gpu=self.cfg.data.get('workers_per_gpu', 1)),
                **dict(samples_per_gpu=self.cfg.data.get('samples_per_gpu', 1)),
                **self.cfg.data.get('test_dataloader', {})
            }
            data_loader = build_dataloader(dataset, **test_loader_cfg)

            # build the model and load checkpoint
            model = build_posenet(self.cfg.model)
            fp16_cfg = self.cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            load_checkpoint(model, self.args.checkpoint, map_location='cpu')

            if self.args.fuse_conv_bn:
                model = fuse_conv_bn(model)

            if not distributed:
                model = MMDataParallel(model, device_ids=[self.args.gpu_id])
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
                outputs = multi_gpu_test(model, data_loader, self.args.tmpdir, self.args.gpu_collect)
                fin = time.time()

                tiempo_ejecucion = fin - inicio
                # print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")

            rank, _ = get_dist_info()
            eval_config = self.cfg.get('evaluation', {})
            eval_config = merge_configs(eval_config, dict(metric=self.args.eval))

            if rank == 0:
                if self.args.out:
                    print(f'\nwriting results to {self.args.out}')
                    mmcv.dump(outputs, self.args.out)
                # tst.get_measure_info(self.cfg.ann_file_measure, outputs)
                results = dataset.evaluate(outputs, self.cfg.work_dir, err_dis=True, **eval_config)
                pd_mae, gt_mae = create_test_quantitative_results(outputs, dataset, checkpoint_name, self.cfg, external_test=external)
                if self.args.predict_images and not complete:
                    create_test_qualitative_images(outputs, dataset, checkpoint_name, self.cfg)
                # tst.save_error_per_point_histogram(results['PCKdis'], os.path.join(self.args.work_dir, "1_ERR_DIS_NEW_" + checkpoint_name + ".png"), metric="px")
                del results['PCKdis']
                results['mae_pd_rm'] = pd_mae
                results['mae_gt_rm'] = gt_mae
                create_custom_file(os.path.join(self.cfg.work_dir, "_test_quantitative_" + checkpoint_name + ".txt"),
                                   results)

        if external:
            check_dict = {}
            for checkpoint in checkpoints:
                checkpoint_name = os.path.basename(checkpoint)[:-4]
                file = os.path.join(self.cfg.work_dir, "_test_quantitative_" + checkpoint_name + ".txt")
                with open(file, 'r') as f:
                    for line in f:
                        if line.startswith('EPE:'):
                            valor_epe = float(line.split(':')[1].strip())
                            check_dict[checkpoint_name] = valor_epe
                            break

            winner_checkpoint_name = min(check_dict, key=check_dict.get)
            return os.path.join(self.cfg.work_dir, "_test_qualitative_" + winner_checkpoint_name)
