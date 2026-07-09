# Copyright (c) Phigent Robotics. All rights reserved.

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import model.networks as networks
from model.base_model import BaseModel
from collections import OrderedDict
from util.util import YCrCb2RGB

class Diffusion_Fusion_Model(BaseModel):
    def __init__(self,opt,local_rank):
        super(Diffusion_Fusion_Model, self).__init__(opt)
        # define Network
        self.Fusion_net = networks.define_Network(opt,local_rank)
        self.local_rank=local_rank
        self.schedule_phase = None
        self.centered = opt['datasets']['centered']
        self.best_metric = None
        self.best_epoch = -1

        # set loss and load resume state
        self.set_loss()

        if self.opt['phase'] == 'train':
            self.Fusion_net.train()
            train_opt = self.opt['train']
            
            optim_params = list(self.Fusion_net.parameters())
            # Set the optimizer
            self.optG = torch.optim.AdamW(optim_params, lr=train_opt["optimizer"]["lr"], betas=(0.9, 0.999),weight_decay=train_opt["optimizer"]["weight_decay"])
            self.optimizers.append(self.optG)
            # Set the learning rate
            self.setup_schedulers()
            self.log_dict = OrderedDict()
        self.load_network()
        # self.print_network(self.Fusion_net)

    def feed_data(self, data):
        for key in data:
            data[key] = data[key].cuda(self.local_rank)
        self.data =data

    def _get_structure_function_inputs(self):
        structure_y = self.data['structure_y']
        function_y = self.data['function_y']
        function_crcb = self.data['function_crcb']
        return structure_y, function_y, function_crcb
        

    def optimize_parameters(self):
    
        self.optG.zero_grad()
        structure_y, function_y, function_crcb = self._get_structure_function_inputs()
        output = self.Fusion_net(structure_y, function_y)
        # Fusion result
        self.Fusion_result = YCrCb2RGB(torch.cat((output['Fusion'], function_crcb[:, 0:1, :, :], function_crcb[:, 1:2, :, :]), dim=1))

        loss_structure = output.get('loss_structure', output['loss_gradient'])
        loss_l1 = output['loss_l1']
        loss_SSIM = output['loss_SSIM']
        dif_loss = output['dif_loss']
      
        loss = output['loss']
        #Averaging loss
        reduce_loss_structure=self.reduce_tensor(loss_structure.data)
        reduce_loss_l1=self.reduce_tensor(loss_l1.data)
        reduce_loss_SSIM=self.reduce_tensor(loss_SSIM.data)
        reduce_dif_loss=self.reduce_tensor(dif_loss.data)
        reduce_loss=self.reduce_tensor(loss.data)

        # Back propagation
        loss.backward()
        self.optG.step()
       
        # Set log
        self.log_dict['l_dif'] = reduce_dif_loss.item()
        self.log_dict['l_ssim'] =reduce_loss_SSIM.item()
        self.log_dict['l_1'] = reduce_loss_l1.item()
        self.log_dict['l_struct'] = reduce_loss_structure.item()
        self.log_dict['l_g'] = reduce_loss_structure.item()
        self.log_dict['l_tot'] = reduce_loss.item()


    def reduce_tensor(self, tensor: torch.Tensor):
        "Average over multiple processes"
        rt = tensor.clone()
        # 仅在分布式训练启用时使用分布式操作
        if self.opt['distributed'] and torch.distributed.is_initialized():
            torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
            rt /= torch.distributed.get_world_size()
        return rt
    
    def test_fusion(self):
        self.Fusion_net.eval()
       
        structure_y, function_y, function_crcb = self._get_structure_function_inputs()
        input = torch.cat([structure_y, function_y], dim=1)
     
        if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
            self.output = self.Fusion_net.module.test_Fusion(input, self.device)
            self.output = YCrCb2RGB(torch.cat((self.output, function_crcb[:, 0:1, :, :], function_crcb[:, 1:2, :, :]), dim=1))
        else:
            self.output = self.Fusion_net.test_Fusion(input, self.device)
            self.output = YCrCb2RGB(torch.cat((self.output, function_crcb[:, 0:1, :, :], function_crcb[:, 1:2, :, :]), dim=1))
        self.Fusion_net.train()

    def validate_fusion(self, deterministic_seed=None):
        was_training = self.Fusion_net.training
        self.Fusion_net.eval()

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        if deterministic_seed is not None:
            torch.manual_seed(int(deterministic_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(deterministic_seed))

        with torch.no_grad():
            structure_y, function_y, function_crcb = self._get_structure_function_inputs()
            output = self.Fusion_net(structure_y, function_y)
            self.output = YCrCb2RGB(torch.cat(
                (output['Fusion'], function_crcb[:, 0:1, :, :], function_crcb[:, 1:2, :, :]),
                dim=1
            ))

            loss_structure = output.get('loss_structure', output['loss_gradient'])
            loss_l1 = output['loss_l1']
            loss_ssim = output['loss_SSIM']
            dif_loss = output['dif_loss']
            loss = output['loss']

            val_logs = OrderedDict()
            val_logs['val_loss'] = self.reduce_tensor(loss.data).item()
            val_logs['val_struct'] = self.reduce_tensor(loss_structure.data).item()
            val_logs['val_l1'] = self.reduce_tensor(loss_l1.data).item()
            val_logs['val_ssim'] = self.reduce_tensor(loss_ssim.data).item()
            val_logs['val_dif'] = self.reduce_tensor(dif_loss.data).item()

        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        if was_training:
            self.Fusion_net.train()
        return val_logs
        
    

    # 设置损失
    def set_loss(self):
        if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
            self.Fusion_net.module.set_loss(self.device)
        else:
            self.Fusion_net.set_loss(self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        structure_rgb = self.data['structure_rgb']
        function_rgb = self.data['function_rgb']
        out_dict['structure'] = self.tensor2im(structure_rgb, min_max=(0, 1))
        out_dict['function'] = self.tensor2im(function_rgb, min_max=(0, 1))
        out_dict['Fusion'] = self.tensor2fu(self.Fusion_result, min_max=(0, 1))

        return out_dict

    def get_current_test(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)

        structure_rgb = self.data['structure_rgb']
        function_rgb = self.data['function_rgb']
        out_dict['structure'] = self.tensor2im(structure_rgb, min_max=(0, 1))
        out_dict['function'] = self.tensor2im(function_rgb, min_max=(0, 1))
        out_dict['Fusion'] = self.tensor2fu(self.output, min_max=(0, 1))
        return out_dict

    def tensor2im(self, image_tensor, imtype=np.float32, min_max=(-1, 1)):
        # (1, 3, 256, 256)===>(3, 256, 256)
        image_numpy = image_tensor[:1, :, :, :].squeeze(0).detach().clamp_(-1, 1).float().cpu().numpy()
        image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        nc, nh, nw = image_numpy.shape

        if nc == 1:
            tmp = np.zeros((nh, nw, 1))
            tmp = image_numpy.transpose(1, 2, 0)
            tmp = np.tile(tmp, (1, 1, 3))
            image_numpy = tmp
        elif nc == 3:
            tmp = np.zeros((nh, nw, 3))
            tmp = image_numpy.transpose(1, 2, 0)
            image_numpy = tmp

        image_numpy -= np.amin(image_numpy)
        #image_numpy = (image_numpy / np.amax(image_numpy))
        image_numpy = (image_numpy /2.0)

        image_numpy = image_numpy * 255.0
        return image_numpy.astype(imtype)

    def tensor2fu(self, image_tensor, imtype=np.float32, min_max=(-1, 1)):
        # (1, 3, 256, 256)===>(3, 256, 256)
        image_numpy = image_tensor[:1, :, :, :].squeeze(0).detach().clamp_(-1, 1).float().cpu().numpy()
        image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  

        nc, nh, nw = image_numpy.shape

        if nc == 1:
            tmp = np.zeros((nh, nw, 1))
            tmp = image_numpy.transpose(1, 2, 0)
            tmp = np.tile(tmp, (1, 1, 3))
            image_numpy = tmp
        elif nc == 3:
            tmp = np.zeros((nh, nw, 3))
            tmp = image_numpy.transpose(1, 2, 0)
            image_numpy = tmp
        image_numpy -= np.amin(image_numpy)
        #image_numpy = (image_numpy / np.amax(image_numpy))
        image_numpy = (image_numpy /2.0)

        image_numpy = image_numpy * 255.0
        return image_numpy.astype(imtype)

    def save_network(self, epoch, iter_step, tag=None, extra_state=None):
        if tag is None:
            genG_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
            opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        else:
            genG_path = os.path.join(self.opt['path']['checkpoint'], '{}_gen_G.pth'.format(tag))
            opt_path = os.path.join(self.opt['path']['checkpoint'], '{}_opt.pth'.format(tag))
        # gen
        network = self.Fusion_net
        if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)

        # opt
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'scheduler': None,
            'optimizer': None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
        }
        opt_state['optimizer'] = self.optG.state_dict()
        if extra_state:
            opt_state.update(extra_state)
        torch.save(opt_state, opt_path)

    @staticmethod
    def _resolve_opt_path(load_path):
        if load_path.endswith('_gen_G.pth'):
            return load_path.replace('_gen_G.pth', '_opt.pth')
        return '{}_opt.pth'.format(load_path)

    @staticmethod
    def _load_state_dict_compat(network, state_dict):
        current_state = network.state_dict()
        filtered_state = OrderedDict()
        skipped_shape = []
        unexpected = []

        for key, value in state_dict.items():
            if key not in current_state:
                unexpected.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            filtered_state[key] = value

        missing = sorted(set(current_state.keys()) - set(filtered_state.keys()))
        network.load_state_dict(filtered_state, strict=False)
        loaded_ratio = len(filtered_state) / max(len(current_state), 1)

        print(
            'Checkpoint compatibility load: loaded={}, missing={}, unexpected={}, shape_mismatch={}, loaded_ratio={:.3f}'.format(
                len(filtered_state), len(missing), len(unexpected), len(skipped_shape), loaded_ratio
            )
        )
        if missing:
            print('Missing sample:', missing[:10])
        if unexpected:
            print('Unexpected sample:', unexpected[:10])
        if skipped_shape:
            print('Shape mismatch sample:', skipped_shape[:10])
        if loaded_ratio < 0.9:
            raise RuntimeError(
                'Checkpoint structure mismatch is too large for safe test-time loading. '
                'Please use a checkpoint and matching config from the same experiment.'
            )


    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            print(load_path)
            if not os.path.isfile(load_path):
                print(f'Warning: resume checkpoint not found, training from scratch: {load_path}')
                return

            genG_path = load_path

            opt_path = self._resolve_opt_path(load_path)
            # gen
            network = self.Fusion_net
            if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
                network = network.module
            state_dict = torch.load(genG_path, map_location='cpu')
            strict_load = not self.opt['model'].get('finetune_norm', False)
            try:
                network.load_state_dict(state_dict, strict=strict_load)
            except RuntimeError:
                if self.opt['phase'] != 'test' and not self.opt['model'].get('finetune_norm', False):
                    raise
                self._load_state_dict_compat(network, state_dict)

            if self.opt['phase'] == 'train':
                if self.opt['model'].get('finetune_norm', False):
                    print('Finetune compatibility mode: optimizer state is reinitialized.')
                    return
                # optimizer
                if os.path.isfile(opt_path):
                    opt = torch.load(opt_path)
                    self.optG.load_state_dict(opt['optimizer'])
                    self.begin_step = opt['iter']
                    self.begin_epoch = opt['epoch']
                    self.best_metric = opt.get('best_metric', self.best_metric)
                    self.best_epoch = opt.get('best_epoch', self.best_epoch)
                else:
                    print(f'Warning: optimizer checkpoint not found, optimizer will be initialized from scratch: {opt_path}')
