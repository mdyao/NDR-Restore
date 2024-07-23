import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
import numpy as np
import time

import random
from collections import OrderedDict
from torch.nn import functional as F
logger = logging.getLogger('base')


class NDR_Model(BaseModel):
    def __init__(self, opt):
        super(NDR_Model, self).__init__(opt)

        self.opt = opt

        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)

        # loss
        self.Rec = ReconstructionLoss(losstype=self.train_opt['pixel_criterion'])

        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    
    @torch.no_grad()
    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.lq = data['lq'].to(self.device)
     
    def feed_data_test(self, data):
        self.gt = data['gt'].to(self.device)  
        self.lq = data['lq'].to(self.device)  

    def feed_data_demo(self, data):
        self.lq = data['lq'].to(self.device)    
        self.name_lq = data['name_lq']

    def loss_calculation(self,x, y):
        loss = self.Rec(x, y)
        return loss

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        self.hq_fake = self.netG.module.restoration_process(self.lq)
        self.lq_fake = self.netG.module.degradation_process(self.gt)

        l_degradation = self.loss_calculation(self.lq_fake, self.lq)
        l_restoration = self.loss_calculation(self.hq_fake, self.gt)
        loss = self.train_opt['lambda_degrad'] * l_degradation + self.train_opt['lambda_restore'] * l_restoration 

        # backward
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_degradation'] = l_degradation.item()
        self.log_dict['l_restoration'] = l_restoration.item()

    def test(self):

        self.netG.eval()
        with torch.no_grad():
            self.out_img = self.netG.module.restoration_process(self.lq)

        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['out_img'] = self.out_img.detach()[0].float().cpu()
        out_dict['lq_img'] = self.lq.detach()[0].float().cpu()
        out_dict['gt_img'] = self.gt.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_demo(self):
        out_dict = OrderedDict()
        out_dict['out_img'] = self.out_img.detach()[0].float().cpu()
        out_dict['lq_img'] = self.lq.detach()[0].float().cpu()
        out_dict['name_lq'] = self.name_lq
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):

        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
