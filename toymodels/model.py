"""Toy models."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class ToyModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
    ):
        super(ToyModel, self).__init__()

        self.W = Parameter(torch.empty((n_hidden, n_features)))
        self.b = Parameter(torch.empty(n_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        hidden = self.W @ x
        return self.W.T @ hidden + self.b, hidden
