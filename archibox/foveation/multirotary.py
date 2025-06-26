import torch
import torch.nn as nn
import torch.nn.functional as F
from bnnp.nn import FusedLinear, MLPBlock, RMSNorm


class MultiPositionRotary(nn.Module):
    def __init__(
        self,
        head_dim: int,
        pos_dim: int,
        min_freq: float,
        max_freq: float,
        frozen: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.pos_dim = pos_dim
        assert head_dim % 2 == 0

        nfreqs = head_dim // 2
        freqs = torch.randn(nfreqs, pos_dim)
        freqs = freqs / (freqs.pow(2).mean(dim=1, keepdim=True) + 1e-7).sqrt()
        freqs = freqs * (
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, nfreqs)
        ).unsqueeze(-1)
        self.freqs_FP = nn.Parameter(freqs, requires_grad=not frozen)

    def forward(self, x_NTHD: torch.Tensor, pos_NTP: torch.Tensor):
        """
        H: nheads
        P: pos_dim
        D: head_dim
        F: nfreqs == head_dim // 2
        """
        assert x_NTHD.size(-1) == self.head_dim
        assert pos_NTP.size(-1) == self.pos_dim

        theta_NTF = (self.freqs_FP * pos_NTP.unsqueeze(-2)).mean(dim=-1)
        cos_NTF = torch.cos(theta_NTF)
        sin_NTF = torch.sin(theta_NTF)

        x_NTHF, y_NTHF = x_NTHD.float().chunk(2, dim=-1)
        x_out_NTHF = x_NTHF * cos_NTF[..., None, :] - y_NTHF * sin_NTF[..., None, :]
        y_out_NTHF = x_NTHF * sin_NTF[..., None, :] + y_NTHF * cos_NTF[..., None, :]
        return torch.cat([x_out_NTHF, y_out_NTHF], dim=-1).type_as(x_NTHD)


class MultiRotaryDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        head_dim: int,
        rotary: MultiPositionRotary,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.nheads = dim // head_dim

        self.norm = RMSNorm(dim, affine=False)
        self.qkv_weight = nn.Parameter(torch.randn(3, dim, dim) / dim**0.5 / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.rotary = rotary
        self.o_proj = FusedLinear(dim, dim, scale=True, zero_init=True)

    def forward(self, input_NTD, pos_NTP):
        N, T, _ = input_NTD.shape
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_NTD)
        q_NTHD, k_NTHD, v_NTHD = (
            (self.norm(input_NTD) @ qkv_weight.t())
            .view(N, T, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q_NTHD, k_NTHD = self.qk_norm(q_NTHD), self.qk_norm(k_NTHD)
        if self.rotary is not None:
            q_NTHD = self.rotary(q_NTHD, pos_NTP)
            k_NTHD = self.rotary(k_NTHD, pos_NTP)
        x_NHTD = F.scaled_dot_product_attention(
            q_NTHD.transpose(1, 2),
            k_NTHD.transpose(1, 2),
            v_NTHD.transpose(1, 2),
            is_causal=True,
        )
        x_NTD = x_NHTD.transpose(1, 2).flatten(2, 3)
        return input_NTD + self.o_proj(x_NTD)

    # def predict(
    #     self, input_ND, past_kv: tuple[torch.Tensor, torch.Tensor] | None, t: int
    # ):
    #     N, _ = input_ND.shape
    #     qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_ND)
    #     q_Nhd, k_Nhd, v_Nhd = (
    #         (self.norm(input_ND) @ qkv_weight.t())
    #         .view(N, 3 * self.nheads, self.head_dim)
    #         .chunk(3, dim=-2)
    #     )
    #     q_Nhd, k_Nhd = self.qk_norm(q_Nhd), self.qk_norm(k_Nhd)
    #     if self.rotary is not None:
    #         q_Nhd = self.rotary.rotate_single(q_Nhd, t)
    #         k_Nhd = self.rotary.rotate_single(k_Nhd, t)
    #     if past_kv is not None:
    #         past_k_NhTd, past_v_NhTd = past_kv
    #         k_NhTd = torch.cat([past_k_NhTd, k_Nhd.unsqueeze(2)], dim=2)
    #         v_NhTd = torch.cat([past_v_NhTd, v_Nhd.unsqueeze(2)], dim=2)
    #     else:
    #         k_NhTd = k_Nhd.unsqueeze(2)
    #         v_NhTd = v_Nhd.unsqueeze(2)
    #     x_Nh1d = F.scaled_dot_product_attention(q_Nhd.unsqueeze(2), k_NhTd, v_NhTd)
    #     x_ND = x_Nh1d.flatten(1, 3)
    #     return input_ND + self.o_proj(x_ND), (
    #         k_NhTd[:, :, -self.window_size + 1 :],
    #         v_NhTd[:, :, -self.window_size + 1 :],
    #     )


class MultiRotaryDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        head_dim: int,
        pos_dim: int,
        depth: int,
        seq_len: int,
        use_rope: bool,
        min_freq: float,
        max_freq: float,
        frozen_rope: bool = True,
    ):
        super().__init__()
        self.rotary = (
            MultiPositionRotary(
                head_dim=head_dim,
                pos_dim=pos_dim,
                min_freq=min_freq,
                max_freq=max_freq,
                frozen=frozen_rope,
            )
            if use_rope
            else None
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                MultiRotaryDecoderLayer(
                    dim=dim,
                    mlp_dim=mlp_dim,
                    head_dim=head_dim,
                    rotary=self.rotary,
                )
            )
            self.layers.append(MLPBlock(dim=dim, mlp_dim=mlp_dim))

    def forward(self, x_NTD, pos_NP):
        for layer in self.layers:
            if isinstance(layer, MultiRotaryDecoderLayer):
                x_NTD = layer(x_NTD, pos_NP)
            else:
                x_NTD = layer(x_NTD)
        return x_NTD

    # def predict(
    #     self, x_ND, cache: tuple[list[tuple[torch.Tensor, torch.Tensor]], int] | None
    # ):
    #     """Inference (one input token, optional kv_cache). Returns new kv_cache"""
    #     if cache is None:
    #         past_key_values = [None for _ in range(len(self.layers))]
    #         t = 0
    #     else:
    #         past_key_values, t = cache
    #     new_key_values = []

    #     for layer, past_kv in zip(self.layers, past_key_values):
    #         if isinstance(layer, DecoderAttentionLayer):
    #             x_ND, new_kv = layer.predict(x_ND, past_kv, t)
    #             new_key_values.append(new_kv)
    #         else:
    #             x_ND = layer(x_ND)
    #             new_key_values.append(None)
    #     return x_ND, (new_key_values, t + 1)
