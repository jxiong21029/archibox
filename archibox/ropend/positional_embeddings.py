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
            ],
            dim=-1,
        )
        self.register_buffer("embed_HWD", emb)

    def forward(self, input_NHWD: torch.Tensor) -> torch.Tensor:
        return input_NHWD + self.embed_HWD.type_as(input_NHWD)


class AxialRoPEConfig(BaseModel):
    head_dim: int
    min_freq: float
    max_freq: float
    n_zero_freqs: int = 0

    variant: Literal["axial_rotary"] = "axial_rotary"


class AxialRoPE(nn.Module):
    """
    Nearly equivalent to UniformRoPE with direction_spacing == pi/2, but with the exact
    same set of frequencies are used for x's and y's (whereas uniform RoPE would be off
    by one).
    """

    def __init__(self, cfg: AxialRoPEConfig, nh: int, nw: int):
        super().__init__()
        assert cfg.head_dim % 4 == 0
        self.cfg = cfg

        xlim, ylim = math.sqrt(nw / nh), math.sqrt(nh / nw)
        x_HW = torch.linspace(-xlim, xlim, nw).reshape(1, nw).expand(nh, nw)
        y_HW = torch.linspace(-ylim, ylim, nh).reshape(nh, 1).expand(nh, nw)

        n_freqs = cfg.head_dim // 4
        assert 0 <= cfg.n_zero_freqs <= n_freqs
        omega = torch.cat(
            (
                torch.zeros(cfg.n_zero_freqs),
                cfg.min_freq
                * (cfg.max_freq / cfg.min_freq)
                ** torch.linspace(0, 1, n_freqs - cfg.n_zero_freqs),
            )
        )
        # TODO: change to x and then y
        theta_HWF = torch.cat(
            (y_HW.unsqueeze(-1) * omega, x_HW.unsqueeze(-1) * omega), dim=-1
        )
        self.register_buffer("cos_HW1F", torch.cos(theta_HWF).unsqueeze(-2))
        self.register_buffer("sin_HW1F", torch.sin(theta_HWF).unsqueeze(-2))

    def resize_to(self, size: int | tuple[int, int], orig_size: int | tuple[int, int]):
        cfg = self.cfg

        orig_size = (orig_size, orig_size) if isinstance(orig_size, int) else orig_size
        orig_nh, orig_nw = self.cos_HWhF.shape[:2]
        assert orig_size[0] // orig_nh == orig_size[1] // orig_nw
        patch_size = orig_size[0] // orig_nh

        if isinstance(size, int):
            size = (size, size)

        nh, nw = size[0] // patch_size, size[1] // patch_size
        ylim = size[0] / orig_size[0] * math.sqrt(orig_size[0] / orig_size[1])
        xlim = size[1] / orig_size[1] * math.sqrt(orig_size[1] / orig_size[0])
        x_HW = torch.linspace(-xlim, xlim, nw).reshape(1, nw).expand(nh, nw)
        y_HW = torch.linspace(-ylim, ylim, nh).reshape(nh, 1).expand(nh, nw)
        omega = cfg.min_freq * (cfg.max_freq / cfg.min_freq) ** torch.linspace(
            0, 1, cfg.head_dim // 4
        )
        # TODO: change to x and then y to match uniform RoPE
        theta_HWF = torch.cat(
            (y_HW.unsqueeze(-1) * omega, x_HW.unsqueeze(-1) * omega), dim=-1
        )
        self.register_buffer("cos_HW1F", torch.cos(theta_HWF).unsqueeze(-2))
        self.register_buffer("sin_HW1F", torch.sin(theta_HWF).unsqueeze(-2))

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
        x_out_NHWhF = x_NHWhF * self.cos_HW1F - y_NHWhF * self.sin_HW1F
        y_out_NHWhF = x_NHWhF * self.sin_HW1F + y_NHWhF * self.cos_HW1F
        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)


class LieREConfig(BaseModel):
    head_dim: int
    variant: Literal["liere"] = "liere"


class LieRE(nn.Module):
    def __init__(self, cfg: LieREConfig, nh: int, nw: int):
        super().__init__()
        self.cfg = cfg
        self.emb_2dd = nn.Parameter(
            torch.rand(2, cfg.head_dim, cfg.head_dim) * 2 * math.pi
        )
        self.emb_2dd._is_embed = True

        xlim, ylim = math.sqrt(nw / nh), math.sqrt(nh / nw)
        x_HW = torch.linspace(-xlim, xlim, nw).reshape(1, nw).expand(nh, nw)
        y_HW = torch.linspace(-ylim, ylim, nh).reshape(nh, 1).expand(nh, nw)
        positions_HW211 = torch.stack((y_HW, x_HW), dim=-1).reshape(nh, nw, 2, 1, 1)
        self.register_buffer("positions_HW211", positions_HW211)

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        N, H, W, h, d = input_NHWhd.shape
        upper_tri_2dd = torch.triu(self.emb_2dd, diagonal=1) - 0.5  # ?
        skew_bases_2dd = upper_tri_2dd - upper_tri_2dd.transpose(-2, -1)
        generator_pos_HWdd = (self.positions_HW211 * skew_bases_2dd).sum(dim=-3)
        rotation_HWdd = torch.matrix_exp(generator_pos_HWdd)
        assert rotation_HWdd.shape == (H, W, d, d)
        output_NHWhd = torch.matmul(input_NHWhd, rotation_HWdd.type_as(input_NHWhd))
        assert output_NHWhd.shape == input_NHWhd.shape
        return output_NHWhd


class UniformRoPEConfig(BaseModel):
    n_heads: int
    head_dim: int
    min_freq: float
    max_freq: float
    n_zero_freqs: int = 0
    direction_spacing: float | None = math.pi * (1 - math.sqrt(5))
    learnable: bool = False

    variant: Literal["uniform_rotary"] = "uniform_rotary"


class UniformRoPE(nn.Module):
    def __init__(self, cfg: UniformRoPEConfig, nh: int, nw: int):
        super().__init__()
        self.cfg = cfg

        n_freqs = cfg.head_dim // 2
        assert 0 <= cfg.n_zero_freqs <= n_freqs
        omega_F = torch.cat(
            (
                torch.zeros(cfg.n_zero_freqs),
                cfg.min_freq
                * (cfg.max_freq / cfg.min_freq)
                ** torch.linspace(0, 1, n_freqs - cfg.n_zero_freqs),
            ),
        )
        if cfg.direction_spacing is not None:
            phi_hF = (
                torch.arange(cfg.n_heads * n_freqs).reshape(cfg.n_heads, n_freqs)
                * cfg.direction_spacing
            )
        else:
            phi_hF = torch.rand((cfg.n_heads, n_freqs)) * 2 * torch.pi
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2

        xlim, ylim = math.sqrt(nw / nh), math.sqrt(nh / nw)
        x_HW = torch.linspace(-xlim, xlim, nw).reshape(1, nw).expand(nh, nw)
        y_HW = torch.linspace(-ylim, ylim, nh).reshape(nh, 1).expand(nh, nw)
        # TODO: should be x then y since phi is cos then sin
        positions_HW112 = torch.stack((y_HW, x_HW), dim=-1).reshape(nh, nw, 1, 1, 2)

        if cfg.learnable:
            self.freqs_hF2 = nn.Parameter(freqs_hF2)
            self.freqs_hF2._is_embed = True
            self.register_buffer("positions_HW112", positions_HW112)
        else:
            theta_HWhF = (freqs_hF2 * positions_HW112).sum(dim=-1)
            self.register_buffer("cos_HWhF", torch.cos(theta_HWhF))
            self.register_buffer("sin_HWhF", torch.sin(theta_HWhF))

    def resize_to(self, size: int | tuple[int, int], orig_size: int | tuple[int, int]):
        cfg = self.cfg

        orig_size = (orig_size, orig_size) if isinstance(orig_size, int) else orig_size
        if cfg.learnable:
            orig_nh, orig_nw = self.positions_HW112.shape[:2]
        else:
            orig_nh, orig_nw = self.cos_HWhF.shape[:2]
        assert orig_size[0] // orig_nh == orig_size[1] // orig_nw
        patch_size = orig_size[0] // orig_nh

        if isinstance(size, int):
            size = (size, size)

        nh, nw = size[0] // patch_size, size[1] // patch_size
        ylim = size[0] / orig_size[0] * math.sqrt(orig_size[0] / orig_size[1])
        xlim = size[1] / orig_size[1] * math.sqrt(orig_size[1] / orig_size[0])

        yy_HW = torch.linspace(-ylim, ylim, nh).reshape(nh, 1).expand(nh, nw)
        xx_HW = torch.linspace(-xlim, xlim, nw).reshape(1, nw).expand(nh, nw)
        positions_HW112 = torch.stack((yy_HW, xx_HW), dim=-1).reshape(nh, nw, 1, 1, 2)
        if cfg.learnable:
            device, dtype = self.positions_HW112.device, self.positions_HW112.dtype
            self.register_buffer("positions_HW112", positions_HW112.to(device, dtype))
        else:
            n_freqs = cfg.head_dim // 2
            omega_F = cfg.min_freq * (cfg.max_freq / cfg.min_freq) ** torch.linspace(
                0, 1, n_freqs
            )
            if cfg.direction_spacing is not None:
                phi_hF = (
                    torch.arange(cfg.n_heads * n_freqs).reshape(cfg.n_heads, n_freqs)
                    * cfg.direction_spacing
                )
            else:
                phi_hF = torch.rand((cfg.n_heads, n_freqs)) * 2 * torch.pi
            directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
            freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2

            theta_HWhF = (freqs_hF2 * positions_HW112).sum(dim=-1)
            device, dtype = self.cos_HWhF.device, self.cos_HWhF.dtype
            self.register_buffer("cos_HWhF", torch.cos(theta_HWhF).to(device, dtype))
            self.register_buffer("sin_HWhF", torch.sin(theta_HWhF).to(device, dtype))

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)

        if self.cfg.learnable:
            theta_HWhF = (self.freqs_hF2.float() * self.positions_HW112.float()).sum(
                dim=-1
            )
            cos_HWhF = torch.cos(theta_HWhF)
            sin_HWhF = torch.sin(theta_HWhF)
            x_out_NHWhF = x_NHWhF * cos_HWhF - y_NHWhF * sin_HWhF
            y_out_NHWhF = x_NHWhF * sin_HWhF + y_NHWhF * cos_HWhF
        else:
            x_out_NHWhF = x_NHWhF * self.cos_HWhF - y_NHWhF * self.sin_HWhF
            y_out_NHWhF = x_NHWhF * self.sin_HWhF + y_NHWhF * self.cos_HWhF

        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)
