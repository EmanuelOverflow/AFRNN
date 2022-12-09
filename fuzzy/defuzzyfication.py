import torch
import torch.nn as nn


class DenseDefuzzy(nn.Module):
    def __init__(self, in_features: int, out_features: int, normalization: bool = True,
                 kernel_initializer: nn.init = nn.init.xavier_normal_,
                 bias: bool = True, kernel_constraint=None):
        super(DenseDefuzzy, self).__init__()
        self.normalization = normalization
        self.function = nn.Linear(in_features, out_features, bias=bias)
        self.kernel_constraint = kernel_constraint

        kernel_initializer(self.function.weight)

    def forward(self, inputs: torch.Tensor):
        logits = self.function(inputs)
        return logits if not self.normalization else \
            logits / (torch.sum(inputs, dim=-1, keepdim=True) + 1e-10)
