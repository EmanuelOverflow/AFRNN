#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:44:55 2020

@author: emanueldinardo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import nn_utils
import math


class MaxMin2d(nn.Module):
    def __init__(self, in_channels, out_fuzzy_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1,
                 bias: bool = True, kernel_constraint=None, padding=0):
        super(MaxMin2d, self).__init__()
        self.in_channels = in_channels
        self.out_fuzzy_channels = out_fuzzy_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.kernel_constraint = kernel_constraint

        self.weight = nn_utils.create_parameter_with_initializer((in_channels * kernel_size ** 2, out_fuzzy_channels),
                                                                 nn.init.uniform_)
        if bias:
            self.bias = nn_utils.create_parameter_with_initializer((out_fuzzy_channels, ), nn.init.uniform_)
        else:
            self.register_parameter('bias', None)

    def _op(self, x):
        weight = self.weight.transpose(0, 1).reshape(1, 1, -1)
        x = x.transpose(-1, -2)
        x_tile = x.repeat(1, 1, weight.shape[-1] // x.shape[-1])
        min_els = torch.where(x_tile < weight, x_tile, weight)
        min_els = min_els.view(-1, x.shape[1], weight.shape[-1] // x.shape[-1], x.shape[-1])
        return min_els.max(-1)[0].transpose(-1, -2)

    def forward(self, x):
        # if x is (B, 3, 32, 32) then x_unf, with kernel_size=3, is (B, 27, 900)
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride,
                         dilation=self.dilation, padding=self.padding)
        x_spatial_size = int(math.sqrt(x_unf.shape[-1]))
        print(x_unf.shape)
        out = self._op(x_unf)
        # Fold patches
        out = out.reshape(-1, self.out_fuzzy_channels, x_spatial_size, x_spatial_size)

        if self.bias:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return 'in_channels={}, out_fuzzy_channels={}, bias={}'.format(
            self.in_channels, self.out_fuzzy_channels, self.bias is not None
        )


class MaxYager2d(nn.Module):
    def __init__(self, in_channels, out_fuzzy_channels: int, kernel_size: int, p: float = 1., stride: int = 1,
                 dilation: int = 1, bias: bool = True, kernel_constraint=None, padding=0):
        super(MaxYager2d, self).__init__()
        self.in_channels = in_channels
        self.out_fuzzy_channels = out_fuzzy_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.kernel_constraint = kernel_constraint

        self.weight = nn_utils.create_parameter_with_initializer((in_channels * kernel_size ** 2, out_fuzzy_channels),
                                                                 nn.init.uniform_)
        if bias:
            self.bias = nn_utils.create_parameter_with_initializer((out_fuzzy_channels, ), nn.init.uniform_)
        else:
            self.register_parameter('bias', None)

        if p == -1:
            self.p = nn_utils.create_parameter_with_initializer((1,), nn.init.ones_)
        else:
            self.p = nn.Parameter(torch.tensor(p), requires_grad=False)

    def _op(self, x):
        # print("P", self.p.data)
        weight = self.weight.transpose(0, 1).reshape(1, 1, -1)
        x = x.transpose(-1, -2)
        x_tile = x.repeat(1, 1, weight.shape[-1] // x.shape[-1])
        if self.p == 0.:
            els = torch.where(x_tile < weight, x_tile, weight)
        elif torch.isinf(self.p):
            els = torch.where(x_tile < weight, x_tile, weight)
        else:
            x1 = (1 - x_tile) ** self.p
            y1 = (1 - weight) ** self.p
            # els = 1 - ((1 - x_tile) ** self.p + (1 - weight) ** self.p) ** (1 / self.p)
            els1 = x1 + y1
            els2 = els1 ** (1 / self.p)
            els = 1 - els2

            # els = 1 - ((1 - x_tile) ** self.p + (1 - weight) ** self.p) ** (1 / self.p)
            els = torch.where(els > 0., els, torch.zeros_like(els))
        els = els.view(-1, x.shape[1], weight.shape[-1] // x.shape[-1], x.shape[-1])
        return els.max(-1)[0].transpose(-1, -2)

    def forward(self, x):
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride,
                         dilation=self.dilation, padding=self.padding)
        x_spatial_size = int(math.sqrt(x_unf.shape[-1]))
        out = self._op(x_unf)
        # Fold patches
        # out = F.fold(x)
        out = out.reshape(-1, self.out_fuzzy_channels, x_spatial_size, x_spatial_size)

        if self.bias:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return 'in_channels={}, out_fuzzy_channels={}, p={}, bias={}'.format(
            self.in_channels, self.out_fuzzy_channels, float(self.p), self.bias is not None
        )


class LeakyThreshold(nn.Module):
    def __init__(self, alpha=0.15, threshold=0.5):
        super(LeakyThreshold, self).__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, x):
        return torch.where(x >= self.threshold, x, self.alpha * x)
