import math
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel


class AbsolutePEConfig(BaseModel):
    dim: int
    init_std: float = 0.05

    variant: Literal["absolute"] = "absolute"


class AbsolutePE(nn.Module):
    def __init__(self, cfg: AbsolutePEConfig, nh: int, nw: int):
        super().__init__()
        self.cfg = cfg
        self.embed_HWD = nn.Parameter(torch.randn(nh, nw, cfg.dim) * cfg.init_std)
        self.embed_HWD._is_embed = True

    def forward(self, input_NHWD: torch.Tensor) -> torch.Tensor:
        return input_NHWD + self.embed_HWD.type_as(input_NHWD)


class FixedSinCosPEConfig(BaseModel):
    dim: int
    min_freq: float
    max_freq: float

    variant: Literal["fixed"] = "fixed"


class FixedSinCosPE(nn.Module):
    def __init__(self, cfg: FixedSinCosPEConfig, nh: int, nw: int):
        super().__init__()
        assert cfg.dim % 4 == 0
        self.cfg = cfg
        n_freqs = cfg.dim // 4
        omega_F = cfg.min_freq * (cfg.max_freq / cfg.min_freq) ** torch.linspace(
            0, 1, n_freqs
        )
        y_HW = torch.linspace(-1, 1, nh).reshape(nh, 1).expand(nh, nw)
        x_HW = torch.linspace(-1, 1, nw).reshape(1, nw).expand(nh, nw)
        emb = torch.cat(
            [
                torch.cos(y_HW.unsqueeze(-1) * omega_F),
                torch.sin(y_HW.unsqueeze(-1) * omega_F),
                torch.cos(x_HW.unsqueeze(-1) * omega_F),
                torch.sin(x_HW.unsqueeze(-1) * omega_F),
            ]
        )
        self.register_buffer("embed_HWD", emb)

    def forward(self, input_NHWD: torch.Tensor) -> torch.Tensor:
        return input_NHWD + self.embed_HWD.type_as(input_NHWD)


class AxialRoPEConfig(BaseModel):
    head_dim: int
    min_freq: float
    max_freq: float

    variant: Literal["axial_rotary"] = "axial_rotary"


class AxialRoPE(nn.Module):
    """
    Nearly equivalent to UniformRoPE with direction_spacing == pi/2, but the exact same
    set of frequencies are used for x's and y's (whereas uniform RoPE would be off by
    one).
    """

    def __init__(self, cfg: AxialRoPEConfig, nh: int, nw: int):
        super().__init__()

        assert cfg.head_dim % 4 == 0
        omega = cfg.min_freq * (cfg.max_freq / cfg.min_freq) ** torch.linspace(
            0, 1, cfg.head_dim // 4
        )
        y_HW = torch.linspace(-1, 1, nh).reshape(nh, 1).expand(nh, nw)
        x_HW = torch.linspace(-1, 1, nw).reshape(1, nw).expand(nh, nw)
        positions_HW12 = torch.stack((y_HW, x_HW), dim=-1).reshape(nh, nw, 1, 2)
        theta_HWF = (
            torch.cat((omega, omega)).reshape(cfg.head_dim // 2, 1) * positions_HW12
        ).mean(dim=-1)
        self.register_buffer("cos_HW1F", torch.cos(theta_HWF).unsqueeze(-2))
        self.register_buffer("sin_HW1F", torch.cos(theta_HWF).unsqueeze(-2))

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
        x_out_NHWhF = x_NHWhF * self.cos_HW1F - y_NHWhF * self.sin_HW1F
        y_out_NHWhF = x_NHWhF * self.sin_HW1F + y_NHWhF * self.cos_HW1F
        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)


class UniformRoPEConfig(BaseModel):
    head_dim: int
    min_freq: float
    max_freq: float
    direction_spacing: float | None = math.pi * (1 - math.sqrt(5))

    variant: Literal["uniform_rotary"] = "uniform_rotary"


class UniformRoPE(nn.Module):
    def __init__(self, cfg: UniformRoPEConfig, nh: int, nw: int):
        super().__init__()

        n_freqs = cfg.head_dim // 2
        freqs_F = cfg.min_freq * (cfg.max_freq / cfg.min_freq) ** torch.linspace(
            0, 1, n_freqs
        )
        if cfg.direction_spacing is not None:
            phi_F = torch.arange(n_freqs) * cfg.direction_spacing
        else:
            phi_F = torch.rand(n_freqs) * 2 * torch.pi
        u_F2 = torch.stack((torch.cos(phi_F), torch.sin(phi_F)), dim=-1)

        y = torch.linspace(-1, 1, nh).reshape(nh, 1).expand(nh, nw)
        x = torch.linspace(-1, 1, nw).reshape(1, nw).expand(nh, nw)
        positions_HW12 = torch.stack((y, x), dim=-1).reshape(nh, nw, 1, 2)
        theta_HWF = (u_F2 * freqs_F.reshape(n_freqs, 1) * positions_HW12).mean(dim=-1)
        self.register_buffer("cos_HW1F", torch.cos(theta_HWF).unsqueeze(-2))
        self.register_buffer("sin_HW1F", torch.sin(theta_HWF).unsqueeze(-2))

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
        x_out_NHWhF = x_NHWhF * self.cos_HW1F - y_NHWhF * self.sin_HW1F
        y_out_NHWhF = x_NHWhF * self.sin_HW1F + y_NHWhF * self.cos_HW1F
        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)


class MixedRoPEConfig(BaseModel):
    head_dim: int
    min_freq: float
    max_freq: float
    learnable: bool = True


class MixedRoPE(nn.Module):
    def __init__(self, cfg: MixedRoPEConfig):
        pass
