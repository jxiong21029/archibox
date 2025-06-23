import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


class ReLU2(nn.Module):
    def forward(self, x: Tensor):
        return F.relu(x).square()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine=False):
        super().__init__()
        self.dim = dim
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor):
        assert x.size(-1) == self.dim, (
            f"expected x.size(-1) == {self.dim} but got {x.size(-1)=}"
        )
        return F.rms_norm(x.float(), (x.size(-1),), self.weight).type_as(x)


class FusedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale: bool = False,
        zero_init: bool = False,
        gain: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features) * gain
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        if scale:
            self.scale = nn.Parameter(torch.ones(out_features))
            if zero_init:
                self.scale.data.zero_()
        else:
            self.register_parameter("scale", None)
            if zero_init:
                self.weight.data.zero_()

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


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, activation=ReLU2):
        super().__init__()
        self.mlp = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, mlp_dim),
            activation(),
            FusedLinear(mlp_dim, dim, zero_init=True, scale=True),
        )

    def forward(self, x):
        return x + self.mlp(x)


def make_embedding(num_embeddings: int, embedding_dim: int, dtype=None):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    if dtype is not None:
        embed = embed.to(dtype)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


def make_output_head(dim: int, out_dim: int):
    head = nn.Sequential(
        RMSNorm(dim, affine=False), FusedLinear(dim, out_dim, zero_init=True)
    )
    head[-1].weight._is_output = True
    return head


def pad_to_multiple(n, k=64):
    return ((n + k - 1) // k) * k


class SoftmaxHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, pad_to: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.head = make_output_head(dim, pad_to_multiple(n_classes, pad_to))

    def forward(self, x):
        return self.head(x)[..., : self.n_classes]

    def loss(self, x, labels, reduction="mean"):
        assert x.shape[:-1] == labels.shape
        logits = self(x).float()
        return F.cross_entropy(
            logits.flatten(0, -2), labels.flatten(), reduction=reduction
        )

    def sample(self, x, sample_shape=()):
        logits = self(x).float()
        return Categorical(logits=logits).sample(sample_shape)


class Rotary(nn.Module):
    def __init__(self, seq_len: int, head_dim: int, base: float):
        super().__init__()
        self.seq_len = seq_len

        freqs = (1 / base) ** torch.linspace(0, 1, head_dim // 4, dtype=torch.float32)
        # Used during inference
        self.freqs_d = nn.Buffer(
            torch.cat([freqs, torch.zeros_like(freqs)]), persistent=False
        )
        theta_1T1d = (torch.arange(seq_len).unsqueeze(-1) * self.freqs_d).reshape(
            1, seq_len, 1, head_dim // 2
        )
        # Used during training
        self.cos_1T1d = nn.Buffer(torch.cos(theta_1T1d), persistent=False)
        self.sin_1T1d = nn.Buffer(torch.sin(theta_1T1d), persistent=False)

    def forward(self, x_NThd):
        """training rotary embedding (known sequence length, uses cached cos/sin)"""
        assert x_NThd.size(1) == self.seq_len
        x1_NThd, x2_NThd = x_NThd.float().chunk(2, dim=-1)
        y1_NThd = x1_NThd * self.cos_1T1d + x2_NThd * self.sin_1T1d
        y2_NThd = x1_NThd * (-self.sin_1T1d) + x2_NThd * self.cos_1T1d
        return torch.cat([y1_NThd, y2_NThd], dim=-1).type_as(x_NThd)

    def rotate_single(self, x_Nhd, t: int):
        """inference rotary embedding (one input token)"""
        theta_d = t * self.freqs_d
        cos_d = torch.cos(theta_d)
        sin_d = torch.sin(theta_d)
        x1_Nhd, x2_Nhd = x_Nhd.float().chunk(2, dim=-1)
        y1_Nhd = x1_Nhd * cos_d + x2_Nhd * sin_d
        y2_Nhd = x1_Nhd * (-sin_d) + x2_Nhd * cos_d
        return torch.cat([y1_Nhd, y2_Nhd], dim=-1).type_as(x_Nhd)


class DecoderAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        head_dim: int,
        rotary: Rotary | None,
        window_size: int,
        block_mask: BlockMask,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.nheads = dim // head_dim
        self.window_size = window_size

        self.norm = RMSNorm(dim, affine=False)
        self.qkv_weight = nn.Parameter(torch.randn(3, dim, dim) / dim**0.5 / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.rotary = rotary
        self.block_mask = block_mask
        self.o_proj = FusedLinear(dim, dim, scale=True, zero_init=True)

    def forward(self, input_NTD):
        N, T, _ = input_NTD.shape
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_NTD)
        q_NThd, k_NThd, v_NThd = (
            (self.norm(input_NTD) @ qkv_weight.t())
            .view(N, T, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q_NThd, k_NThd = self.qk_norm(q_NThd), self.qk_norm(k_NThd)
        if self.rotary is not None:
            q_NThd, k_NThd = self.rotary(q_NThd), self.rotary(k_NThd)
        x_NhTd = flex_attention(
            q_NThd.transpose(1, 2),
            k_NThd.transpose(1, 2),
            v_NThd.transpose(1, 2),
            block_mask=self.block_mask,
        )
        x_NTD = x_NhTd.transpose(1, 2).flatten(2, 3)
        return input_NTD + self.o_proj(x_NTD)

    def predict(
        self, input_ND, past_kv: tuple[torch.Tensor, torch.Tensor] | None, t: int
    ):
        N, _ = input_ND.shape
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_ND)
        q_Nhd, k_Nhd, v_Nhd = (
            (self.norm(input_ND) @ qkv_weight.t())
            .view(N, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q_Nhd, k_Nhd = self.qk_norm(q_Nhd), self.qk_norm(k_Nhd)
        if self.rotary is not None:
            q_Nhd = self.rotary.rotate_single(q_Nhd, t)
            k_Nhd = self.rotary.rotate_single(k_Nhd, t)
        if past_kv is not None:
            past_k_NhTd, past_v_NhTd = past_kv
            k_NhTd = torch.cat([past_k_NhTd, k_Nhd.unsqueeze(2)], dim=2)
            v_NhTd = torch.cat([past_v_NhTd, v_Nhd.unsqueeze(2)], dim=2)
        else:
            k_NhTd = k_Nhd.unsqueeze(2)
            v_NhTd = v_Nhd.unsqueeze(2)
        x_Nh1d = F.scaled_dot_product_attention(q_Nhd.unsqueeze(2), k_NhTd, v_NhTd)
        x_ND = x_Nh1d.flatten(1, 3)
        return input_ND + self.o_proj(x_ND), (
            k_NhTd[:, :, -self.window_size + 1 :],
            v_NhTd[:, :, -self.window_size + 1 :],
        )


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        head_dim: int,
        depth: int,
        seq_len: int,
        window_size: int,
        use_rope: bool,
        rope_base: float,
    ):
        super().__init__()
        self.rotary = (
            Rotary(seq_len=seq_len, head_dim=head_dim, base=rope_base)
            if use_rope
            else None
        )

        def sliding_window_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx < kv_idx + window_size)

        block_mask = create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                DecoderAttentionLayer(
                    dim=dim,
                    mlp_dim=mlp_dim,
                    head_dim=head_dim,
                    rotary=self.rotary,
                    window_size=window_size,
                    block_mask=block_mask,
                )
            )
            self.layers.append(MLPBlock(dim=dim, mlp_dim=mlp_dim))

    def forward(self, x_NTD):
        """Training forward (fixed seq_len, no kv_cache)"""
        for layer in self.layers:
            x_NTD = layer(x_NTD)
        return x_NTD

    def predict(
        self, x_ND, cache: tuple[list[tuple[torch.Tensor, torch.Tensor]], int] | None
    ):
        """Inference (one input token, optional kv_cache). Returns new kv_cache"""
        if cache is None:
            past_key_values = [None for _ in range(len(self.layers))]
            t = 0
        else:
            past_key_values, t = cache
        new_key_values = []

        for layer, past_kv in zip(self.layers, past_key_values):
            if isinstance(layer, DecoderAttentionLayer):
                x_ND, new_kv = layer.predict(x_ND, past_kv, t)
                new_key_values.append(new_kv)
            else:
                x_ND = layer(x_ND)
                new_key_values.append(None)
        return x_ND, (new_key_values, t + 1)
