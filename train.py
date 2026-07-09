import os

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
import random
from util.visualizer import Visualizer
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train_pet_mri.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--sample_selected', type=str, default=None, help='select diffusion sampler')
    parser.add_argument('--model_selected', type=str, default=None, help='select diffusion network')
    parser.add_argument('--batch_size', type=int, default=None, help='set batch_size')
    parser.add_argument('--fusion_task', type=str, default=None, help='set fusion_task')
    parser.add_argument('--strategy', type=str, default=None, help='set fusion strategy')
    parser.add_argument('--max_steps', type=int, default=None, help='stop after this many training steps for smoke tests')
    #=====================================================================
    #           First：Set parameters
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    # 处理local_rank参数
    if args.local_rank is None:
        args.local_rank = 0
    
    # 只在命令行参数不为None时才覆盖配置文件
    if args.sample_selected is not None:
        opt['model']['Fusion']['sample_selected']=args.sample_selected
    if args.model_selected is not None:
        opt['model']['Fusion']['model_selected']=args.model_selected
    if args.batch_size is not None:
        opt['datasets']['train']['batch_size']=args.batch_size
    if args.fusion_task is not None:
        opt['model']['fusion_task']=args.fusion_task
    if args.strategy is not None:
        opt['model']['Fusion']['mode']=args.strategy
    visualizer = Visualizer(opt)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Set up the training logger
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    ##################################################
    #                  Fix Random Seed
    #################################################
    seed = opt['seed'] if 'seed' in opt else 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed: {seed}")
    ##################################################
    #              Initialize distributed operations
    #################################################
    gpus=len(opt['gpu_ids'])
    if opt['distributed']:
        dist.init_process_group(backend='gloo',world_size=gpus,rank=args.local_rank)
        torch.cuda.set_device(args.local_rank)
    else:
        # Single GPU training
        torch.cuda.set_device(0)
    ##################################################
    #             Read dataset
    #################################################
    phase = 'train'
    dataset_opt=opt['datasets']['train']
    batchSize = dataset_opt['batch_size']
    
    # 检查医学图像数据路径
    if args.local_rank == 0:
        print(f"Checking medical image dataset paths...")
        structure_path = dataset_opt.get('dataroot_structure', '')
        function_path = dataset_opt.get('dataroot_function', '')
        if not os.path.exists(structure_path):
            print(f"WARNING: structure dataset path does not exist: {structure_path}")
        if not os.path.exists(function_path):
            print(f"WARNING: function dataset path does not exist: {function_path}")
    
    train_set = Data.create_dataset_Fusion(dataset_opt, phase,opt)
    val_loader = None
    val_set = None
    val_dataset_opt = opt['datasets'].get('val')
    if phase == 'train':
        if opt['distributed']:
            train_sampler=DistributedSampler(train_set,shuffle=True)  #Random sampling
            train_loader =torch.utils.data.DataLoader(
                train_set,
                batch_size=batchSize,
                shuffle=dataset_opt['use_shuffle'],               
                sampler=train_sampler,
                num_workers=dataset_opt['num_workers'],            #Multiple threads to load data for efficiency
                pin_memory=True,                                   #Load data into GPU memory to speed up training
            )
        else:
            # Single GPU training - 不使用分布式采样器
            train_loader =torch.utils.data.DataLoader(
                train_set,
                batch_size=batchSize,
                shuffle=dataset_opt['use_shuffle'],               
                num_workers=dataset_opt['num_workers'],            #Multiple threads to load data for efficiency
                pin_memory=True,                                   #Load data into GPU memory to speed up training
            )
        if val_dataset_opt is not None:
            if args.local_rank == 0:
                val_structure_path = val_dataset_opt.get('dataroot_structure', '')
                val_function_path = val_dataset_opt.get('dataroot_function', '')
                if not os.path.exists(val_structure_path):
                    print(f"WARNING: val structure dataset path does not exist: {val_structure_path}")
                if not os.path.exists(val_function_path):
                    print(f"WARNING: val function dataset path does not exist: {val_function_path}")

            val_set = Data.create_dataset_Fusion(val_dataset_opt, 'val', opt)
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=val_dataset_opt.get('batch_size', 1),
                shuffle=False,
                num_workers=val_dataset_opt.get('num_workers', 0),
                pin_memory=True,
            )
    training_iters = int(ceil(train_set.length / float(batchSize*gpus)))
    #Set the cosine annealing learning rate
    scheduler_type = opt['train']['scheduler']['type']
    if scheduler_type == 'CosineAnnealingRestartCyclicLR' and 'periods' in opt['train']['scheduler']:
        original_list = opt['train']['scheduler']['periods']
        result_list = [x * training_iters for x in original_list]
        opt['train']['scheduler']['periods'] = result_list
    elif scheduler_type == 'MultiStepLR' and 'milestones' in opt['train']['scheduler']:
        # MultiStepLR milestones are in epochs, no need to multiply by training_iters
        pass
    else:
        # For other schedulers like StepLR, no special processing needed
        pass
    if args.local_rank==0:
        logger.info('Initial Dataset Finished')
    # Instantiation model
    diffusion = Model.create_model(opt,args.local_rank)
    if args.local_rank == 0:
        logger.info('Initial Model Finished')
   

    ################################################################
    ###                          train                            ###
    ################################################################
    current_step = diffusion.begin_step
    start_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    val_epoch_freq = int(opt['train'].get('val_epoch_freq', 1))
    best_metric_name = opt['train'].get('save_best_metric', 'val_loss')
    best_mode = str(opt['train'].get('save_best_mode', 'min')).lower()
    val_seed = int(opt['train'].get('val_seed', 1234))
    best_metric = diffusion.best_metric if diffusion.best_metric is not None else (float('inf') if best_mode == 'min' else -float('inf'))
    early_stop_opt = opt['train'].get('early_stop', {}) or {}
    early_stop_enabled = bool(early_stop_opt.get('enabled', False))
    early_stop_patience = int(early_stop_opt.get('patience', 12))
    early_stop_min_delta = float(early_stop_opt.get('min_delta', 0.0))
    early_stop_start_epoch = int(early_stop_opt.get('start_epoch', 0))
    early_stop_bad_epochs = 0
    if opt['path']['resume_state']:
        if args.local_rank == 0:
            print('Resuming training from epoch: {}, iter: {}.'.format(start_epoch, current_step))

    for current_epoch in range (start_epoch,n_epoch):
        if args.local_rank == 0:
            print(f"Starting epoch {current_epoch}/{n_epoch}")

        if opt['distributed']:
            train_sampler.set_epoch(current_epoch)     #Make the data random
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            if args.local_rank == 0 and istep % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"Epoch {current_epoch}, Step {istep+1}/{training_iters} | GPU: {gpu_memory:.1f}GB/{gpu_cached:.1f}GB")
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            diffusion.update_learning_rate(current_epoch*train_set.length+istep*batchSize, warmup_iter=opt['train'].get('warmup_iter', -1))
            if args.max_steps is not None and current_step >= args.max_steps:
                if args.local_rank == 0:
                    print(f"Reached --max_steps={args.max_steps}; stopping short training run.")
                break

            #
            #          logging
            if args.local_rank == 0:
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    lr_log=diffusion.get_current_learning_rate()
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, lr_log[0], 'Train')
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)
        if args.max_steps is not None and current_step >= args.max_steps:
            break

        if val_loader is not None and (current_epoch + 1) % val_epoch_freq == 0:
            val_logs_sum = {}
            last_visuals = None
            for val_idx, val_data in enumerate(val_loader):
                diffusion.feed_data(val_data)
                batch_val_logs = diffusion.validate_fusion(deterministic_seed=val_seed + val_idx)
                for key, value in batch_val_logs.items():
                    val_logs_sum[key] = val_logs_sum.get(key, 0.0) + float(value)
                if args.local_rank == 0 and (val_idx == 0 or val_idx == len(val_loader) - 1):
                    last_visuals = diffusion.get_current_test()

            mean_val_logs = {
                key: value / max(len(val_loader), 1)
                for key, value in val_logs_sum.items()
            }

            if args.local_rank == 0:
                lr_log = diffusion.get_current_learning_rate()
                visualizer.print_current_errors(current_epoch, len(val_loader), len(val_loader), mean_val_logs, lr_log[0], 'Val')
                if last_visuals is not None:
                    visualizer.display_current_results(last_visuals, current_epoch, True)

                current_metric = mean_val_logs.get(best_metric_name)
                if current_metric is None:
                    logger.warning(f'Validation metric "{best_metric_name}" not found. Skipping best-model update.')
                else:
                    if best_mode == 'min':
                        improved = current_metric < (best_metric - early_stop_min_delta)
                    else:
                        improved = current_metric > (best_metric + early_stop_min_delta)
                    if improved:
                        best_metric = current_metric
                        early_stop_bad_epochs = 0
                        diffusion.best_metric = current_metric
                        diffusion.best_epoch = current_epoch
                        diffusion.save_network(
                            current_epoch,
                            current_step,
                            tag='best',
                            extra_state={
                                'best_metric_name': best_metric_name,
                                'best_metric_value': current_metric,
                            }
                        )
                        logger.info(
                            'Best model updated at epoch %d: %s=%.6f',
                            current_epoch,
                            best_metric_name,
                            current_metric
                        )
                        print(f'Best model updated at epoch {current_epoch}: {best_metric_name}={current_metric:.6f}')
                    elif early_stop_enabled and current_epoch >= early_stop_start_epoch:
                        early_stop_bad_epochs += 1
                        logger.info(
                            'Early stopping counter: %d/%d (%s=%.6f, best=%.6f)',
                            early_stop_bad_epochs,
                            early_stop_patience,
                            best_metric_name,
                            current_metric,
                            best_metric
                        )
                        if early_stop_bad_epochs >= early_stop_patience:
                            logger.info(
                                'Early stopping triggered at epoch %d. Best epoch=%d, best %s=%.6f',
                                current_epoch,
                                diffusion.best_epoch,
                                best_metric_name,
                                best_metric
                            )
                            print(
                                f'Early stopping triggered at epoch {current_epoch}. '
                                f'Best epoch={diffusion.best_epoch}, {best_metric_name}={best_metric:.6f}'
                            )
                            break

        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            if args.local_rank==0:
                diffusion.save_network(current_epoch, current_step)
    # 只在分布式训练时清理进程组
    if opt['distributed'] and torch.distributed.is_initialized():
        dist.destroy_process_group()
