#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:54:03 2020

@author: emanueldinardo
"""

import torch
import torch.nn as nn
from .fuzzy import nn_utils


class Gaussian1D(nn.Module):
    """
    Never tested
    """
    def __init__(self, in_channels, truth_degrees):
        self.truth_degrees = truth_degrees
        self.c = nn_utils.create_parameter_with_initializer((in_channels, self.truth_degrees), nn.init.uniform_)
        self.a = nn_utils.create_parameter_with_initializer((in_channels, self.truth_degrees), nn.init.ones_)
        super(Gaussian1D, self).__init__()

    def call(self, x):
        xc = torch.exp(-torch.square((x - self.c) / (2 * torch.square(self.a))))
        return xc


class MinMaxMembership(nn.Module):
    def __init__(self, min_value=0,  max_value=1):
        self.min_value = min_value
        self.max_value = max_value
        super(MinMaxMembership, self).__init__()

    def call(self, x):
        scaled = (x - x.min()) / (x.max() - x.min())
        v = scaled * (self.max_value - self.min_value) + self.min_value
        return v


class ScalingMembership(nn.Module):
    def __init__(self, scale_factor=255.):
        self.scale_factor = scale_factor
        super(ScalingMembership, self).__init__()

    def call(self, x):
        return x / self.scale_factor
