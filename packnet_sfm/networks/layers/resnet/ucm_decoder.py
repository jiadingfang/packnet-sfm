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
        self.convs['linear_focal'] = nn.Linear(int(128 * 128 / 1024), 2)
        self.convs['linear_offset'] = nn.Linear(int(128 * 128 / 1024), 2)
        self.convs['linear_alpha'] = nn.Linear(int(128 * 128 / 1024), 1)
        # self.convs['linear_focal'] = nn.Linear(int(192 * 640 / 1024), 2)
        # self.convs['linear_offset'] = nn.Linear(int(192 * 640 / 1024), 2)
        # self.convs['linear_alpha'] = nn.Linear(int(192 * 640 / 1024), 1)
        self.convs['softplus'] = nn.Softplus()
        self.convs['sigmoid'] = nn.Sigmoid()

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.tanh = nn.Tanh()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        x = self.convs['conv1'](x).squeeze()
        x = torch.flatten(x)

        f = self.convs['linear_focal'](x)
        f = self.convs['softplus'](f)
        fx = f[0] * 128
        fy = f[1] * 128

        c = self.convs['linear_offset'](x)
        cx = (c[0] + 0.5) * 128
        cy = (c[1] + 0.5) * 128

        # f = self.convs['linear_focal'](x)
        # f = self.convs['softplus'](f)
        # fx = f[0] * 192
        # fy = f[1] * 640

        # c = self.convs['linear_offset'](x)
        # cx = (c[0] + 0.5) * 192
        # cy = (c[1] + 0.5) * 640

        alpha = self.convs['linear_alpha'](x)
        # print(alpha)
        alpha = self.convs['sigmoid'](alpha)
        # print(alpha)

        # fx = 128 / 2
        # fy = 128 / 2
        # cx = 128 / 2
        # cy = 128 / 2
        # alpha = 0

        I = torch.tensor([fx, fy, cx, cy, alpha])

        # print('fx = {}'.format(fx))
        # print('fy = {}'.format(fy))
        # print('cx = {}'.format(cx))
        # print('cy = {}'.format(cy))
        # print('alpha = {}'.format(alpha))

        self.output = I

        return self.output
