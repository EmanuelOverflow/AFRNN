#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:32:38 2020

@author: angelociaramella
"""

import torch
import torch.nn as nn


# Definizione del MinLayer

# class FuzzyMinimum(nn.Module):
#     def __init__(self, dim):
#         self.dim = dim
#         super(FuzzyMinimum, self).__init__()
#
#     def call(self, inputs):
#         if type(inputs) == list:
#             inputs = torch.stack(inputs, dim=self.dim)
#         out = torch.min(inputs, dim=self.dim)
#         return out


class GlobalMinPooling2D(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(GlobalMinPooling2D, self).__init__()

    def call(self, x):
        return torch.min(x, dim=self.dim)
