import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
import torch.distributed as dist

import numpy as np
from app.utils import *
from app.models.baseModel import baseModel

from ptflops import get_model_complexity_info
import importlib

class Model(baseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.ws = torch.distributed.get_world_size() if config.dist else 1
        self.device = config.device

        ### NETWORKS ###
        ## main network
        lib = importlib.import_module('models.archs.{}'.format(config.network))
        if self.rank <= 0: print(toGreen('Loading Model...'))
        self.network = DeblurNet(config, lib).to(self.device)

        ### PROFILE ###
        if self.rank <= 0:
            with torch.no_grad():
                Macs, params = get_model_complexity_info(
                    self.network.Network, (1, 3, 720, 1280), input_constructor=self.network.input_constructor,
                    as_strings=False, print_per_layer_stat=config.is_verbose
                    )

        ### DDP ###
        if config.cuda:
            self.network = DP(self.network).to(torch.device('cuda'))

            if self.rank <= 0:
                print(toGreen('Computing model complexity...'))
                print(toRed('\t{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3)))
                print(toRed('\t{:<30}  {:<8} M'.format('Number of parameters: ', params / 1000 ** 2, '\n')))


class DeblurNet(nn.Module):
    def __init__(self, config, lib):
        super(DeblurNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1

        self.config = config
        self.device = config.device
        self.Network = lib.Network(config)
        if self.rank <= 0: print(toRed('\tinitializing deblurring network'))

    def input_constructor(self, res):
        b, c, h, w = res[:]

        C = torch.FloatTensor(np.random.randn(b, c, h, w)).to(self.device)

        return {'C': C, 'R': C, 'L': C}

    #####################################################
    def forward(self, C, R=None, L=None, GT=None, is_train=False):
        is_train = is_train or self.config.save_sample and self.config.is_train

        outs = self.Network(C, R, L, is_train)

        return outs
