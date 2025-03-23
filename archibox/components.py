import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU2(nn.Module):
    def forward(self, x: Tensor):
        return F.relu(x).square()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool):
        super().__init__()
        self.dim = dim
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor):
        assert x.size(-1) == self.dim
        return F.rms_norm(x.float(), (x.size(-1),), self.weight).type_as(x)


class FusedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=False, scale=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features) / 2
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        if scale:
            self.scale = nn.Parameter(torch.ones(out_features))
        else:
            self.register_parameter("scale", None)

    def forward(self, x: Tensor):
        if self.scale is not None:
            # Fused per-channel scaling
            weight = self.weight * self.scale.unsqueeze(1)
        else:
            weight = self.weight

        out = x @ weight.type_as(x).t()

        if self.bias is not None:
            out = out + self.bias.type_as(out)
        return out
