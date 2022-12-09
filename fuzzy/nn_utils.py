import torch
import torch.nn as nn


def create_parameter_with_initializer(size, initializer, params=None):
    param = nn.Parameter(torch.empty(size))
    if params is None:
        initializer(param)
    else:
        initializer(param, *params)
    return param
