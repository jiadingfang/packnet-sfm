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
    def __init__(self, init_intrinsic, num_ch_enc, scales=[0], num_output_channels=3, use_skips=True):
        super(UCMDecoder, self).__init__()

        # camera intrinsic parameter as a vector
        if len(init_intrinsic) > 0:
            i = torch.tensor(init_intrinsic)
            i[:4] = i[:4] / 1000.0
            sigmoid_inv_i = torch.log(i / (1 - i))
            self.intrinsic_vector = nn.Parameter(sigmoid_inv_i)
        else:
            self.intrinsic_vector = nn.Parameter(-torch.ones(5))
            
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.output = {}

        # get forcal length and offsets
        x = input_features[-1]
        B = x.shape[0]
        
        # single dataset tensor
        fx, fy, cx, cy = self.sigmoid(self.intrinsic_vector[0:4]) * 1000
        alpha = self.sigmoid(self.intrinsic_vector[4]) * 1.0

        I = torch.zeros(5)
        I[0] = fx
        I[1] = fy
        I[2] = cx
        I[3] = cy
        I[4] = alpha

        self.output = I.unsqueeze(0).repeat(B,1)

        return self.output
