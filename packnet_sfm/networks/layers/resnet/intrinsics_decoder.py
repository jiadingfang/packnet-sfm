# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock, Conv3x3, upsample


class IntrinsicsDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=[0], num_output_channels=3, use_skips=True):
        super(IntrinsicsDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

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

        self.convs['conv1'] = ConvBlock(512, 1)
        self.convs['linear1'] = nn.Linear(int(128 * 128 / 1024), 4)
        # self.convs['linear1'] = nn.Linear(int(192 * 640 / 1024), 4)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.tanh = nn.Tanh()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        x = self.convs['conv1'](x).squeeze()
        # print(x.shape)
        x = torch.flatten(x)
        # print(x.shape)
        x = self.convs['linear1'](x)
        # print(x.shape)

        fx = x[0] * 128
        fy = x[1] * 128
        cx = (x[2] + 0.5) * 128
        cy = (x[3] + 0.5) * 128

        # fx = x[0] * 192
        # fy = x[1] * 640
        # cx = (x[2] + 0.5) * 192
        # cy = (x[3] + 0.5) * 640

        k = torch.eye(3)
        k[0,0] = x[0]
        k[1,1] = x[1]
        k[0,2] = x[2]
        k[1,2] = x[3]
        self.output = k

        # # decoder
        # x = input_features[-1]
        # #print(x.shape)
        # x = self.convs['conv1'](x).squeeze()
        # k = torch.eye(3)
        # k[0,0] = x[0,0]
        # k[1,1] = x[1,1]
        # k[0,2] = x[0,2]
        # k[1,2] = x[1,2]
        # self.output = k

        return self.output
