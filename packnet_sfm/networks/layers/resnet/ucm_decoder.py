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
        # self.intrinsic_vector = nn.Parameter(torch.zeros(5))
        self.intrinsic_vector = nn.Parameter(-torch.ones(5))

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
        # x = self.convs['conv1'](x).squeeze()
        # x = torch.flatten(x)

        # f = self.convs['linear_focal'](x)
        # f = self.convs['softplus'](f)
        # fx = f[0] * 128
        # fy = f[1] * 128

        # c = self.convs['linear_offset'](x)
        # cx = (c[0] + 0.5) * 128
        # cy = (c[1] + 0.5) * 128

        # f = self.convs['linear_focal'](x)
        # f = self.convs['activation_focal'](f)
        # fx = f[0] * 80 + 370
        # fy = f[1] * 80 + 370

        # c = self.convs['linear_offset'](x)
        # c = self.convs['activation_offset'](c)
        # cx = (c[0]) * 80 + 320
        # cy = (c[1]) * 80 + 91

        # f = self.convs['linear_focal'](x)
        # f = self.convs['softplus'](f)
        # fx = f[0] * 384
        # fy = f[1] * 384

        # c = self.convs['linear_offset'](x)
        # cx = (c[0] + 0.5) * 384
        # cy = (c[1] + 0.5) * 384

        # alpha = self.convs['linear_alpha'](x)
        # alpha = 1.0 / 2 * self.convs['activation_alpha'](alpha)

        # fx = 128 / 2
        # fy = 128 / 2
        # cx = 128 / 2
        # cy = 128 / 2
        
        fx, fy, cx, cy = self.sigmoid(self.intrinsic_vector[0:4]) * 1000
        alpha = self.sigmoid(self.intrinsic_vector[4]) * 1 / 2

        I = torch.zeros(5)
        I[0] = fx
        I[1] = fy
        I[2] = cx
        I[3] = cy
        I[4] = alpha
        # I = torch.tensor([fx, fy, cx, cy, alpha])
        # I = torch.tensor([fx, fy, cx, cy, alpha], requires_grad=True)
        # print()
        # print('I')
        # print(I)
        # print(I.requires_grad)

        # print()
        # print('I shape')
        # print(I.shape)
        # print('fx = {}'.format(fx))
        # print('fy = {}'.format(fy))
        # print('cx = {}'.format(cx))
        # print('cy = {}'.format(cy))
        # print('alpha = {}'.format(alpha))

        self.output = I.unsqueeze(0).repeat(B,1)
        # print('output shape')
        # print(self.output.shape)

        return self.output
