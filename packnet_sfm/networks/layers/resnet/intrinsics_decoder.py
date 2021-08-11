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

        # camera intrinsic parameter as a vector
        self.intrinsic_vector = nn.Parameter(torch.zeros(4))

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
        # self.convs['linear_focal'] = nn.Linear(int(192 * 640 * 1 / 1024), 2)
        # self.convs['linear_offset'] = nn.Linear(int(192 * 640 * 1 / 1024), 2)
        # # self.convs['activation_focal'] = nn.Softplus()
        # # self.convs['activation_focal'] = nn.Sigmoid()
        # self.convs['activation_focal'] = nn.Tanh()
        # # self.convs['activation_offset'] = nn.Sigmoid()
        # self.convs['activation_offset'] = nn.Tanh()
        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        B = x.shape[0]
        # x = self.convs['conv1'](x)
        # B = x.size(0)
        # x = x.view(B, -1)

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

        # # fx = 370
        # # fy = 370

        # c = self.convs['linear_offset'](x)
        # c = self.convs['activation_offset'](c)
        # cx = (c[0]) * 80 + 320
        # cy = (c[1]) * 80 + 91

        # # cx = 192 / 2
        # # cy = 640 / 2


        # K = torch.eye(3)
        # batch_K = K.unsqueeze(0).repeat(B,1,1)
        # K[:,0,0] = fx
        # K[:,1,1] = fy
        # K[:,0,2] = cx
        # K[:,1,2] = cy

        # print()
        # print('intrinsic vector')
        # print(self.intrinsic_vecotr.data)
        # print(self.intrinsic_vecotr.grad)

        fx, fy, cx, cy = self.sigmoid(self.intrinsic_vector) * 1000
        K = torch.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy

        # fx, fy, cx, cy = self.sigmoid(torch.zeros(4)) * 1000
        # K = torch.eye(3)
        # K[0,0] = fx
        # K[1,1] = fy
        # K[0,2] = cx
        # K[1,2] = cy
        self.output = K.unsqueeze(0).repeat(B,1,1)

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
