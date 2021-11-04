# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock, Conv3x3, upsample


class UCMDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=[0], num_output_channels=3, use_skips=True):
        super(UCMDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # camera intrinsic parameter as a vector
        # i = torch.tensor([235.4/1000, 245.1/1000, 186.5/1000, 132.6/1000, 0.65])
        # i = i * 0.9
        # i = i * 0.95
        # i = i * 1.05
        # i = i * 1.10
        # sigmoid_inv_i = torch.log(i / (1 - i))
        # self.intrinsic_vector = nn.Parameter(sigmoid_inv_i)
        # self.intrinsic_vector = nn.Parameter(torch.zeros(5))
        self.intrinsic_vector = nn.Parameter(-torch.ones(5))
        # self.intrinsic_vector = nn.Parameter(torch.ones(2,5))
        # self.intrinsic_vector = nn.Parameter(-torch.ones(2,5))
        # self.intrinsic_vector = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0, -2.0, -1.0]))
        # self.intrinsic_vector = nn.Parameter(torch.tensor([[-1.0, -1.0, -1.0, -2.0, -1.0],
        #                                                     [-1.0, -1.0, -1.0, -2.0, -1.0]]))


        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # self.convs['conv1'] = ConvBlock(512, 1)
        # # self.convs['linear_focal'] = nn.Linear(int(128 * 128 / 1024), 2)
        # # self.convs['linear_offset'] = nn.Linear(int(128 * 128 / 1024), 2)
        # # self.convs['linear_alpha'] = nn.Linear(int(128 * 128 / 1024), 1)
        # self.convs['linear_focal'] = nn.Linear(int(192 * 640 / 1024), 2)
        # self.convs['linear_offset'] = nn.Linear(int(192 * 640 / 1024), 2)
        # self.convs['linear_alpha'] = nn.Linear(int(192 * 640 / 1024), 1)
        # # self.convs['linear_focal'] = nn.Linear(int(384 * 384 / 1024), 2)
        # # self.convs['linear_offset'] = nn.Linear(int(384 * 384 / 1024), 2)
        # # self.convs['linear_alpha'] = nn.Linear(int(384 * 384 / 1024), 1)
        # self.convs['activation_focal'] = nn.Tanh()
        # self.convs['activation_offset'] = nn.Tanh()
        # self.convs['activation_alpha'] = nn.Tanh()
        # # self.convs['activation_alpha'] = nn.Sigmoid()

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        B = x.shape[0]
        
        # single dataset tensor
        fx, fy, cx, cy = self.sigmoid(self.intrinsic_vector[0:4]) * 1000
        # alpha = self.sigmoid(self.intrinsic_vector[4]) * 1/2
        alpha = self.sigmoid(self.intrinsic_vector[4]) * 1.0

        I = torch.zeros(5)
        I[0] = fx
        I[1] = fy
        I[2] = cx
        I[3] = cy
        I[4] = alpha

        # I[0] = 235.36
        # I[1] = 245.12
        # I[2] = 186.45
        # I[3] = 132.63
        # I[4] = 0.65

        I[0] = 379.0
        I[1] = 368.0
        I[2] = 314.0
        I[3] = 91.0
        I[4] = 0.0

        self.output = I.unsqueeze(0).repeat(B,1)

        # # double dataset tensor
        # fx_0, fy_0, cx_0, cy_0 = self.sigmoid(self.intrinsic_vector[0, 0:4]) * 1000
        # alpha_0 = self.sigmoid(self.intrinsic_vector[0, 4]) * 1.0

        # fx_1, fy_1, cx_1, cy_1 = self.sigmoid(self.intrinsic_vector[1, 0:4]) * 1000
        # alpha_1 = self.sigmoid(self.intrinsic_vector[1, 4]) * 1.0

        # I = torch.zeros((2,5))
        # I[0,0] = fx_0
        # I[0,1] = fy_0
        # I[0,2] = cx_0
        # I[0,3] = cy_0
        # I[0,4] = alpha_0
        # I[1,0] = fx_1
        # I[1,1] = fy_1
        # I[1,2] = cx_1
        # I[1,3] = cy_1
        # I[1,4] = alpha_1

        # self.output = I.unsqueeze(0).repeat(B,1,1)

        return self.output
