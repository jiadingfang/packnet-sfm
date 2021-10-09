# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock, Conv3x3, upsample


class DSDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=[0], num_output_channels=3, use_skips=True):
        super(DSDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # camera intrinsic parameter as a vector
        # i = torch.tensor([183.85 / 1000, 191.47 / 1000, 186.73 / 1000, 132.81 / 1000, (-0.221 + 1) / 2, 0.576])
        # i = torch.tensor([208.10/1000, 216.78/1000, 186.24/1000, 132.82/1000, (-0.172 + 1)/2, 0.592])
        # i = torch.tensor([181.4/1000, 188.9/1000, 186.4/1000, 132.6/1000, (-0.230+1)/2, 0.571]) # euroc gt
        # i = i * 0.9
        # i = i * 1.10
        # sigmoid_inv_i = torch.log(i / (1 - i))
        # self.intrinsic_vector = nn.Parameter(sigmoid_inv_i)
        # self.intrinsic_vector = nn.Parameter(torch.zeros(6))
        self.intrinsic_vector = nn.Parameter(-torch.ones(6))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        B = x.shape[0]
        
        fx, fy, cx, cy = self.sigmoid(self.intrinsic_vector[0:4]) * 1000
        xi = self.sigmoid(self.intrinsic_vector[4]) * 2 - 1
        alpha = self.sigmoid(self.intrinsic_vector[5]) * 1

        I = torch.zeros(6)
        I[0] = fx
        I[1] = fy
        I[2] = cx
        I[3] = cy
        I[4] = xi
        I[5] = alpha

        self.output = I.unsqueeze(0).repeat(B,1)

        return self.output
