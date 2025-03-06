import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from sandbox.components import FusedLinear, ReLU2, RMSNorm


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.nheads = dim // head_dim

        self.attn_norm = RMSNorm(dim, affine=False)
        self.weight_qkv = nn.Parameter(
            torch.randn(3, dim, dim).fmod(2) / 2 / math.sqrt(dim)
        )
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.o_proj = FusedLinear(dim, dim, scale=True)
        self.o_proj.scale.data.zero_()

        self.mlp = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, 4 * dim),
            ReLU2(),
            FusedLinear(4 * dim, dim, scale=True),
        )
        self.mlp[-1].scale.data.zero_()

    def forward(self, x_NLD: Tensor):
        N, L, _ = x_NLD.shape
        residual = x_NLD

        qkv = self.attn_norm(x_NLD) @ self.weight_qkv.flatten(0, 1).type_as(x_NLD).t()
        q, k, v = qkv.view(N, L, 3 * self.nheads, self.head_dim).chunk(3, dim=-2)
        q, k = self.qk_norm(q), self.qk_norm(k)

        x_NhLd = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        x_NLD = rearrange(
            x_NhLd, "N h L d -> N L (h d)", h=self.nheads, d=self.head_dim
        )
        x_NLD = residual + self.o_proj(x_NLD)

        x_NLD = x_NLD + self.mlp(x_NLD)
        return x_NLD


class VisionTransformer(nn.Module):
    def __init__(
        self, img_height: int, img_width: int, patch_size: int, dim: int, depth: int
    ):
        super().__init__()
        self.patch_size = patch_size

        assert img_height % patch_size == 0
        assert img_width % patch_size == 0
        h = img_height // patch_size
        w = img_width // patch_size
        self.patchify = FusedLinear(patch_size * patch_size * 3, dim)
        self.pos_emb = nn.Parameter(torch.randn(h, w, dim) / 2 / math.sqrt(dim))
        self.pos_emb._ortho = False

        self.layers = nn.Sequential(
            *[EncoderLayer(dim, head_dim=128) for _ in range(depth)]
        )
        self.out_norm = RMSNorm(dim, affine=False)

    def forward(self, x_NCHW: Tensor):
        assert torch.is_floating_point(x_NCHW)

        x_NHWC = rearrange(
            x_NCHW,
            "N C (h1 h2) (w1 w2) -> N h1 w1 (h2 w2 C)",
            h2=self.patch_size,
            w2=self.patch_size,
            C=3,
        )
        x_NHWD = self.patchify(x_NHWC) + self.pos_emb.type_as(x_NHWC)
        x_NLD = x_NHWD.flatten(1, 2)

        x_NLD = self.layers(x_NLD)
        x_ND = self.out_norm(x_NLD).mean(dim=1)
        return x_ND
