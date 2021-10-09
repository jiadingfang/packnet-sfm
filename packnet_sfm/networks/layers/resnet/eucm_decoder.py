# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock, Conv3x3, upsample


class EUCMDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=[0], num_output_channels=3, use_skips=True):
        super(EUCMDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # camera intrinsic parameter as a vector
        # i = torch.tensor([235.64381137951174 / 1000, 245.38803860055288 / 1000, 186.44431894063212 / 1000, 132.64829510142745 / 1000, 0.5966287792627975 / 1, 1.1122253956511319 / 2])
        # i = torch.tensor([251.34/1000, 261.84/1000, 186.08/1000, 132.6/1000, 0.608, 1.082/2])
        # i = torch.tensor([280/1000, 280/1000, 128/1000, 80/1000, 1, 1])
        # i = i * 0.9
        # i = i * 0.95
        # i = i * 1.05
        # i = i * 1.10
        # sigmoid_inv_i = torch.log(i / (1 - i))
        # self.intrinsic_vector = nn.Parameter(sigmoid_inv_i)
        # self.intrinsic_vector = nn.Parameter(torch.zeros(6))
        self.intrinsic_vector = nn.Parameter(-torch.ones(6))

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        B = x.shape[0]
        
        fx, fy, cx, cy = self.sigmoid(self.intrinsic_vector[0:4]) * 1000
        alpha = self.sigmoid(self.intrinsic_vector[4]) * 1/2
        beta = self.sigmoid(self.intrinsic_vector[5]) * 2

        I = torch.zeros(6)
        I[0] = fx
        I[1] = fy
        I[2] = cx
        I[3] = cy
        I[4] = alpha
        I[5] = beta

        self.output = I.unsqueeze(0).repeat(B,1)

        return self.output
