import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")
flex_attention = torch.compile(flex_attention, dynamic=False)


def make_embedding(num_embeddings: int, embedding_dim: int, dtype=None):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    if dtype is not None:
        embed.to(dtype=dtype)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


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


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        seq_len: int,
        window_size: int,
        head_dim: int,
        use_rope: bool,
        rope_base: float,
        scale: bool,
    ):
        assert head_dim % 4 == 0
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len  # training sequence length
        self.window_size = window_size  # max number of tokens to attend to
        self.use_rope = use_rope  # whether to use rotary positional embeddings
        self.rope_base = rope_base  # maximum period of rotary / 2pi
        self.head_dim = head_dim
        self.nheads = dim // head_dim

        self.attn_norm = RMSNorm(dim, affine=False)
        self.qkv_w = nn.Parameter(torch.randn(3, dim, dim) / math.sqrt(dim) / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.o_proj = FusedLinear(dim, dim, scale=scale, zero_init=True)

        if use_rope:
            freqs = (1 / rope_base) ** torch.linspace(
                0, 1, head_dim // 4, dtype=torch.float32
            )
            # Used during inference
            self.freqs_d = nn.Buffer(
                torch.cat((freqs, freqs.new_zeros(head_dim // 4))), persistent=False
            )
            theta_1T1d = torch.outer(torch.arange(seq_len), self.freqs_d).reshape(
                1, seq_len, 1, head_dim // 2
            )
            # Used during training
            self.cos_1T1d = nn.Buffer(torch.cos(theta_1T1d), persistent=False)
            self.sin_1T1d = nn.Buffer(torch.sin(theta_1T1d), persistent=False)

        def sliding_window_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx < kv_idx + window_size)

        self.block_mask = create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
        )

        self.mlp = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, mlp_dim),
            ReLU2(),
            FusedLinear(mlp_dim, dim, scale=scale, zero_init=True),
        )

    def rotary(self, x_NTHD):
        """Training rotary embedding (known seq_len, cached cos/sin)"""
        assert x_NTHD.ndim == 4
        assert x_NTHD.size(1) == self.seq_len
        x1_NTHd, x2_NTHd = x_NTHD.float().chunk(2, dim=-1)
        y1_NTHd = x1_NTHd * self.cos_1T1d + x2_NTHd * self.sin_1T1d
        y2_NTHd = x1_NTHd * (-self.sin_1T1d) + x2_NTHd * self.cos_1T1d
        return torch.cat([y1_NTHd, y2_NTHd], dim=-1).type_as(x_NTHD)

    def inference_rotary(self, x_NHD, t: int):
        """Inference rotary embedding (one input token)"""
        assert x_NHD.ndim == 3
        assert t >= 0
        theta_d = t * self.freqs_d
        cos_d = torch.cos(theta_d)
        sin_d = torch.sin(theta_d)
        x1_NHd, x2_NHd = x_NHD.float().chunk(2, dim=-1)
        y1_NHd = x1_NHd * cos_d + x2_NHd * sin_d
        y2_NHd = x1_NHd * (-sin_d) + x2_NHd * cos_d
        return torch.cat([y1_NHd, y2_NHd], dim=-1).type_as(x_NHD)

    def forward(self, x_NTD: Tensor):
        """Training forward (fixed seq_len, no past_kv)"""
        N, T, _ = x_NTD.shape
        residual = x_NTD
        qkv = self.attn_norm(x_NTD) @ self.qkv_w.flatten(0, 1).type_as(x_NTD).t()
        q_NTHD, k_NTHD, v_NTHD = qkv.view(N, T, 3 * self.nheads, -1).chunk(3, dim=2)
        q_NTHD, k_NTHD = self.qk_norm(q_NTHD), self.qk_norm(k_NTHD)
        if self.use_rope:
            q_NTHD, k_NTHD = self.rotary(q_NTHD), self.rotary(k_NTHD)
        x_NHTD = flex_attention(
            q_NTHD.transpose(1, 2),
            k_NTHD.transpose(1, 2),
            v_NTHD.transpose(1, 2),
            block_mask=self.block_mask,
        )
        x_NTD = x_NHTD.transpose(1, 2).reshape(N, T, self.dim)
        x_NTD = residual + self.o_proj(x_NTD)

        x_NTD = x_NTD + self.mlp(x_NTD)
        return x_NTD

    def predict(
        self, x_ND: Tensor, past_kv: tuple[Tensor, Tensor] | None, t: int | None = None
    ):
        """Inference (one input token, optional past_kv). Returns new kv."""
        N, _ = x_ND.shape
        residual = x_ND
        qkv = self.attn_norm(x_ND) @ self.qkv_w.flatten(0, 1).type_as(x_ND).t()
        q_NHD, k_NHD, v_NHD = qkv.view(N, 3 * self.nheads, -1).chunk(3, dim=1)
        q_NHD, k_NHD = self.qk_norm(q_NHD), self.qk_norm(k_NHD)

        if self.use_rope:
            assert t is not None
            q_NHD = self.inference_rotary(q_NHD, t)
            k_NHD = self.inference_rotary(k_NHD, t)
        else:
            assert t is None

        if past_kv is not None:
            past_k_NHTD, past_v_NHTD = past_kv
            k_NHTD = torch.cat([past_k_NHTD, k_NHD.unsqueeze(2)], dim=2)
            v_NHTD = torch.cat([past_v_NHTD, v_NHD.unsqueeze(2)], dim=2)
        else:
            k_NHTD = k_NHD.unsqueeze(2)
            v_NHTD = v_NHD.unsqueeze(2)
        x_NHTD = F.scaled_dot_product_attention(q_NHD.unsqueeze(2), k_NHTD, v_NHTD)
        x_ND = x_NHTD[:, :, -1, :].flatten(1, 2)
        x_ND = residual + self.o_proj(x_ND)

        x_ND = x_ND + self.mlp(x_ND)
        return x_ND, (
            k_NHTD[:, :, -self.window_size + 1 :],
            v_NHTD[:, :, -self.window_size + 1 :],
        )


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        depth: int,
        seq_len: int,
        window_size: int,
        head_dim: int = 64,
        use_rope: bool = True,
        rope_base: float = 1024.0,
        scale: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=dim,
                    mlp_dim=mlp_dim,
                    seq_len=seq_len,
                    window_size=window_size,
                    head_dim=head_dim,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    scale=scale,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x_NTD: Tensor):
        """Training forward (fixed seq_len, no kv_cache)"""
        for layer in self.layers:
            x_NTD = layer(x_NTD)
        return x_NTD

    def predict(
        self, x: Tensor, kv_cache: list[tuple[Tensor, Tensor]] | None, t: int | None
    ):
        """Inference (one input token, optional kv_cache). Returns new kv_cache"""
        if kv_cache is None:
            kv_cache = [None for _ in range(len(self.layers))]
        new_kv_cache = []

        for layer, past_kv in zip(self.layers, kv_cache, strict=True):
            x, new_kv = layer.predict(x, past_kv, t)
            new_kv_cache.append(new_kv)
        return x, new_kv_cache
