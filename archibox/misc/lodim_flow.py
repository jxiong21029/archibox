import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image
from torchvision.transforms.functional import resize

from archibox.components import FusedLinear, RMSNorm
from archibox.muon import Muon

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


@dataclass
class Config:
    patch_size: int = 8

    dim: int = 1024
    mlp_dim: int = 4096
    depth: int = 4
    freq_dim: int = 256

    n_steps: int = 10_000
    batch_size: int = 8192
    muon_lr: int = 0.01
    adamw_lr: int = 0.001

    n_samples: int = 16384
    sampling_steps: int = 128


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, modulated: bool = False):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.k = FusedLinear(dim, mlp_dim)
        self.v = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)
        if modulated:
            self.wq = FusedLinear(dim, mlp_dim, scale=True, zero_init=True)

    def forward(self, x, c: torch.Tensor | None = None):
        scores = F.relu(self.k(self.norm(x))).pow(2)
        if c is not None:
            scores = scores * (1.0 + self.wq(self.norm(c)))
        return x + self.v(scores)


class MLPFlow(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.freqs = nn.Buffer(torch.pi * (128 ** torch.linspace(0, 1, cfg.freq_dim)))
        self.t_phases = nn.Parameter(torch.randn(cfg.freq_dim))
        self.row_phases = nn.Parameter(torch.randn(cfg.freq_dim))
        self.col_phases = nn.Parameter(torch.randn(cfg.freq_dim))

        input_dim = cfg.patch_size**2 + cfg.freq_dim * 4
        self.in_proj = FusedLinear(input_dim, cfg.dim)
        self.t_encoder = nn.Sequential(
            FusedLinear(cfg.freq_dim * 2, cfg.dim), MLPBlock(cfg.dim, cfg.mlp_dim)
        )
        self.blocks = nn.ModuleList(
            MLPBlock(cfg.dim, cfg.mlp_dim, modulated=True) for _ in range(cfg.depth)
        )
        self.out_proj = nn.Sequential(
            RMSNorm(cfg.dim, affine=False), FusedLinear(cfg.dim, cfg.patch_size**2 + 2)
        )

    def time_embed(self, t):
        assert t.size(-1) == 1
        theta = t.float() * self.freqs + self.t_phases
        return torch.cat([theta.sin(), theta.cos()], dim=-1)

    def forward(self, x_ND, t_N1):
        patch_ND, i_N, j_N = x_ND[:, :-2], x_ND[:, -2], x_ND[:, -1]
        assert patch_ND.size(1) == self.cfg.patch_size**2
        row_ND = i_N.unsqueeze(1).float() * self.freqs + self.row_phases
        col_ND = j_N.unsqueeze(1).float() * self.freqs + self.col_phases
        input_ND = torch.cat(
            [patch_ND, row_ND.sin(), row_ND.cos(), col_ND.sin(), col_ND.cos()], dim=1
        ) * t_N1.pow(0.5)

        x_ND = self.in_proj(input_ND.type(DTYPE))
        c_ND = self.t_encoder(self.time_embed(t_N1).type(DTYPE))
        for block in self.blocks:
            x_ND = block(x_ND, c_ND)
        return self.out_proj(x_ND)

    def sample(self, N: int, n_steps: int):
        device = next(self.parameters()).device
        timesteps = torch.linspace(0, 1, n_steps + 1, device=device).float()
        x = torch.randn(N, self.cfg.patch_size**2 + 2, device=device).float()
        for i in range(n_steps):
            start = timesteps[i].view(1)
            stop = timesteps[i + 1].view(1)

            v_start = self(x, start).float()
            x_stop = x + (stop - start) * v_start
            v_stop = self(x_stop, stop).float()
            x = x + (stop - start) * (v_start + v_stop) / 2
        return x


def sample_patches(image: torch.Tensor, n: int, patch_size: int):
    i = torch.randint(0, image.size(0) - patch_size + 1, (n,))
    j = torch.randint(0, image.size(1) - patch_size + 1, (n,))
    di = torch.arange(patch_size).view(patch_size, 1)
    dj = torch.arange(patch_size).view(1, patch_size)
    patches = image[i[:, None, None] + di, j[:, None, None] + dj]
    assert patches.shape == (n, patch_size, patch_size)
    return i, j, patches


def main():
    logging.basicConfig(level=logging.INFO)

    cfg = Config()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load image and convert to grayscale.
    image = np.array(Image.open(Path(__file__).parent / "peak.jpg").convert("L"))
    image = torch.from_numpy(image).float().cuda() / 255.0 * 2.0 - 1.0
    image = resize(
        image.unsqueeze(0), (image.size(0) // 8, image.size(1) // 8)
    ).squeeze(0)

    model = MLPFlow(cfg)
    model.cuda()
    model = torch.compile(model)

    scalar_params = []
    embeds_params = []
    output_params = []
    muon_params = []
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        if not p.requires_grad:
            log.info(f"{name} {shape} requires_grad=False, skipped")
            continue
        elif p.ndim < 2:
            log.info(f"{name} {shape} assigned to AdamW")
            scalar_params.append(p)
        elif hasattr(p, "_is_embed") and p._is_embed:
            log.info(f"{name} {shape} (_is_embed=True) assigned to AdamW")
            embeds_params.append(p)
        elif hasattr(p, "_is_output") and p._is_output:
            log.info(f"{name} {shape} (_is_output=True) assigned to AdamW")
            output_params.append(p)
        else:
            log.info(f"{name}{shape} assigned to Muon")
            muon_params.append(p)
    total_params = sum(
        p.numel() for p in muon_params + scalar_params + embeds_params + output_params
    )
    total_param_tensors = sum(
        len(group)
        for group in (muon_params, scalar_params, embeds_params, output_params)
    )
    log.info(
        "parameter information:\n"
        f"- muon params: {sum(p.numel() for p in muon_params):,} over {len(muon_params):,} tensors\n"
        f"- scalar params: {sum(p.numel() for p in scalar_params):,} over {len(scalar_params):,} tensors\n"
        f"- embeds params: {sum(p.numel() for p in embeds_params):,} over {len(embeds_params):,} tensors\n"
        f"- output params: {sum(p.numel() for p in output_params):,} over {len(output_params):,} tensors\n"
        f"total: {total_params:,} over {total_param_tensors:,} tensors"
    )
    adamw_params = [
        dict(params=scalar_params, lr=cfg.adamw_lr),
        dict(params=embeds_params, lr=cfg.adamw_lr),
        dict(params=output_params, lr=cfg.adamw_lr),
    ]
    muon = Muon(muon_params, lr=cfg.muon_lr, momentum=0.9, weight_decay=0.01)
    adamw = torch.optim.AdamW(
        adamw_params,
        betas=[0.9, 0.99],
        weight_decay=0.0,
    )

    for group in muon.param_groups:
        group["initial_lr"] = group["lr"]
    for group in adamw.param_groups:
        group["initial_lr"] = group["lr"]

    losses = []
    for t in tqdm.trange(cfg.n_steps, desc="training"):
        y_N, x_N, patches_NHW = sample_patches(
            image, n=cfg.batch_size, patch_size=cfg.patch_size
        )
        patches_ND = patches_NHW.flatten(-2, -1)
        N = patches_ND.size(0)

        y_N1 = y_N.unsqueeze(1).to(patches_ND.device) / (image.size(0) - cfg.patch_size)
        x_N1 = x_N.unsqueeze(1).to(patches_ND.device) / (image.size(1) - cfg.patch_size)
        y_N1, x_N1 = y_N1 * 2.0 - 1.0, x_N1 * 2.0 - 1.0
        data_ND = torch.cat([patches_ND, y_N1, x_N1], dim=1)
        noise_ND = torch.randn_like(data_ND)
        t_N1 = torch.rand(N, 1).cuda()

        input_ND = data_ND * t_N1 + noise_ND * (1 - t_N1)
        vel_ND = data_ND - noise_ND
        pred_ND = model(input_ND, t_N1)

        loss = (pred_ND - vel_ND).pow(2).mean()
        losses.append(loss.detach())

        loss.backward()
        adamw.step()
        muon.step()
        adamw.zero_grad()
        muon.zero_grad()

        frac = t / cfg.n_steps
        relative_lr = 1.0 - (1 - 0.1) * min(1.0, frac)
        for group in muon.param_groups:
            group["lr"] = group["initial_lr"] * relative_lr
        for group in adamw.param_groups:
            group["lr"] = group["initial_lr"] * relative_lr

        if t == 0 or (t + 1) % 1000 == 0:
            log.info(f"{t=} || loss={torch.stack(losses).mean():.4f}")
            losses.clear()
            output = torch.zeros_like(image)
            totals = torch.zeros_like(output)
            for _ in range(cfg.n_samples // cfg.batch_size):
                with torch.inference_mode():
                    samples_ND = model.sample(
                        cfg.batch_size, n_steps=cfg.sampling_steps
                    )
                for sample in samples_ND:
                    patch, y, x = sample[:-2], sample[-2], sample[-1]
                    y = torch.round((y + 1) / 2.0 * image.size(0))
                    x = torch.round((x + 1) / 2.0 * image.size(1))
                    y = y.long().clamp(0, image.size(0) - cfg.patch_size)
                    x = x.long().clamp(0, image.size(1) - cfg.patch_size)
                    output[y : y + cfg.patch_size, x : x + cfg.patch_size] += (
                        patch.view(cfg.patch_size, cfg.patch_size)
                    )
                    totals[y : y + cfg.patch_size, x : x + cfg.patch_size] += 1.0
            output = output / totals.clamp(min=1.0)

            output = (output + 1) / 2.0 * 255.0
            output = output.clamp(0, 255).type(torch.uint8).cpu().numpy()
            Image.fromarray(output).save(
                Path(__file__).parent
                / f"data/output_{t + 1 if t > 0 else 'initial'}.png"
            )

    torch.save(model.state_dict(), Path(__file__).parent / "data/model.ckpt")


if __name__ == "__main__":
    main()
