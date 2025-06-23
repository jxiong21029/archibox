"""
stage 1: train classifier with random patches
- autoregressively decode to a sequence of logits
- loss = mean cross-entropy across all predictions

step 2: learn to select next patch location
- it's probably fine to optimize greedily
- i.e. selection policy should have energy == next nll

Future considerations:
- dense prediction (maybe hard to do efficiently in general)
    - or maybe once this efficient computes a global feature, relatively shallow network
    can decode to dense predictions for each output location
- efficient architecture (e.g. a linear attention variant)
    - or maybe not since sequence lengths are pretty short (probably <1024 for images)

"""

import logging
import os
import random
from pathlib import Path

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange
from pydantic import BaseModel, Field
from torchvision.transforms import v2

from archibox.components import Decoder, FusedLinear, SoftmaxHead
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon, split_muon_adamw_params
from archibox.utils import parse_config

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32
metrics = Metrics(enabled=False, use_wandb=False)


class ModelConfig(BaseModel):
    nlevels: int = 3
    block_size: int = 8  # number of patches to sample at a time
    patch_size: int = 8  # side length of resized patches, in pixels
    seq_len: int = 32
    overlap_patches: bool = True
    detach_patch_head: bool = False

    dim: int = 384
    mlp_dim: int = 768
    depth: int = 4
    nfreqs: int = 64


class Config(BaseModel):
    use_wandb: bool = False
    model: ModelConfig = Field(default_factory=ModelConfig)

    do_compile: bool = False
    n_steps: int = 1000
    valid_every: int | None = 250
    batch_size: int = 1024
    do_augment: bool = True

    muon_lr: float = 0.01
    adamw_lr: float = 0.0005
    wd: float = 0.01
    lr_cooldown_start: int | None = None
    lr_cooldown_ratio: float = 0.0


class FourierEmbed(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nfreqs: int,
        min_freq: float,
        max_freq: float,
        frozen: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim

        z = torch.randn(nfreqs, in_dim)
        z = z / (z.pow(2).mean(dim=1, keepdim=True) + 1e-7).sqrt()
        self.z = nn.Parameter(z, requires_grad=not frozen)
        self.freqs_H = nn.Parameter(
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, nfreqs),
            requires_grad=not frozen,
        )

    def forward(self, x_ND):
        assert x_ND.size(-1) == self.in_dim
        alignments_NH = (self.z * x_ND.unsqueeze(-2)).mean(dim=-1)
        theta_NH = self.freqs_H * alignments_NH
        return torch.cat([torch.sin(theta_NH), torch.cos(theta_NH)], dim=-1)


class PatchDecoder(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        dim: int,
        mlp_dim: int,
        depth: int,
        nfreqs: int,
        seq_len: int,
    ):
        super().__init__()
        self.patchify = FusedLinear(np.prod(patch_size), dim)
        self.pos_embed = FourierEmbed(2, nfreqs, min_freq=0.1, max_freq=100.0)
        self.size_embed = FourierEmbed(1, nfreqs, min_freq=0.1, max_freq=100.0)
        self.emb_proj = FusedLinear(nfreqs * 4, dim)

        self.decoder = Decoder(
            dim,
            mlp_dim,
            head_dim=64,
            depth=depth,
            seq_len=seq_len,
            window_size=seq_len,
            use_rope=False,
            rope_base=0.0,
        )

    def forward(self, patches_NTHWC, pos_NT2, sizes_NT):
        """
        Args:
            patches_NTHWC: patches (should be already resized) extracted from images
            pos_NT2: coordinates of patch centers in [-1, 1].
                -1 is bottom/left, +1 is top/right
            sizes_NT: sizes of patches in [0, 1]
                0 is size 0, 1 is full image

        TODO: replace additive-style positional embedding with proper rotary embeds
        (still from pos_NT2 and not sequence index, so requires custom implementation)
        """
        assert torch.is_floating_point(patches_NTHWC)
        N, T, _, _, _ = patches_NTHWC.shape

        patches_NTD = rearrange(patches_NTHWC, "N T H W C -> N T (H W C)")
        x_NTD = self.patchify(patches_NTD)

        pos_emb = self.pos_embed(pos_NT2).type_as(x_NTD)
        size_emb = self.size_embed(sizes_NT[..., None]).type_as(x_NTD)
        emb_NTD = self.emb_proj(torch.cat([pos_emb, size_emb], dim=-1))

        return self.decoder(x_NTD + emb_NTD)


def build_patch_locs(nlevels: int = 4, overlap: bool = True) -> torch.Tensor:
    """Returns a tensor of shape (total_patches, 4) containing (y_min, x_min, y_max,
    x_max) for every patch in every level.

    Coordinates are in [-1, 1].
    """
    locs = []

    for level in range(nlevels):
        n = 2 ** (level + 1) - 1 if overlap else 2**level
        half_side = 1 / 2**level
        centers = torch.linspace(-1, 1, n + 2)[1:-1]

        cy, cx = torch.meshgrid(centers, centers, indexing="ij")  # (n, n)
        new_locs = torch.stack(
            [cy - half_side, cx - half_side, cy + half_side, cx + half_side], dim=-1
        )  # (n, n, 4)
        locs.append(new_locs.reshape(-1, 4))  # flatten to (n^2, 4)

    return torch.cat(locs, dim=0)


class FoveatedMnist(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.seq_len % cfg.block_size == 0

        self.cfg = cfg
        self.net = PatchDecoder(
            [cfg.patch_size, cfg.patch_size],
            cfg.dim,
            cfg.mlp_dim,
            cfg.depth,
            cfg.nfreqs,
            cfg.seq_len,
        )

        self.patch_locs = nn.Buffer(
            build_patch_locs(nlevels=cfg.nlevels, overlap=cfg.overlap_patches)
        )
        assert -1.0 <= self.patch_locs.amin()
        assert self.patch_locs.amax() <= 1.0

        self.output_head = SoftmaxHead(cfg.dim, 10)
        self.patch_head = SoftmaxHead(cfg.dim, len(self.patch_locs))

    def forward(self, images_NHW, labels_N):
        assert images_NHW.dtype == torch.uint8
        N, H, W = images_NHW.shape
        T, P = self.cfg.seq_len, self.cfg.patch_size
        assert P <= min(W, H)
        dev = images_NHW.device

        images_NHW = images_NHW.type(DTYPE) / 255 * 2 - 1

        # Sample random patch locations (index of upper-left pixel)
        # TODO sample on-policy?
        patch_idx = torch.randint(0, len(self.patch_locs), (N, T), device=dev)
        patch_idx[:, 0] = 0
        y0, x0, y1, x1 = self.patch_locs[patch_idx].unbind(-1)

        ys_NTP = y0[..., None] + (y1 - y0)[..., None] * torch.linspace(0, 1, P).to(dev)
        xs_NTP = x0[..., None] + (x1 - x0)[..., None] * torch.linspace(0, 1, P).to(dev)
        assert ys_NTP.shape == xs_NTP.shape == (N, T, P)
        ys_NTPP = ys_NTP.view(N, T, P, 1).expand(N, T, P, P)
        xs_NTPP = xs_NTP.view(N, T, 1, P).expand(N, T, P, P)
        grid_xy_NTPP2 = torch.stack([xs_NTPP, ys_NTPP], dim=-1)
        assert grid_xy_NTPP2.shape == (N, T, P, P, 2)

        patches_NThw = F.grid_sample(
            einops.repeat(images_NHW, "N H W -> (N T) 1 H W", T=T),
            grid_xy_NTPP2.reshape(N * T, P, P, 2).type(DTYPE),
            mode="bilinear",
            align_corners=False,
        ).reshape(N, T, P, P)
        pos_NT2 = torch.stack([(y0 + y1) / 2, (x0 + x1) / 2], dim=-1)
        sizes_NT = (y1 - y0) / 2

        # Compute class logits
        output_NTD = self.net(patches_NThw.unsqueeze(-1), pos_NT2, sizes_NT)
        class_logits = self.output_head(output_NTD).float()
        class_xent = F.cross_entropy(
            class_logits.flatten(0, -2),
            labels_N.view(N, 1).expand(N, T).flatten(),
            reduction="none",
        ).view(N, T)
        loss = class_xent.mean()

        # Compute patch logits
        if self.cfg.detach_patch_head:
            patch_logits = self.patch_head(output_NTD.detach())[:, :-1]
        else:
            patch_logits = self.patch_head(output_NTD)[:, :-1]
        patch_xent = F.cross_entropy(
            patch_logits.flatten(0, -2),
            patch_idx[:, 1:].flatten(),
            reduction="none",
        ).reshape(N, T - 1)
        with torch.no_grad():
            xent_improvement = -(class_xent[:, 1:] - class_xent[:, :-1]).float()
            mask = xent_improvement > torch.quantile(
                xent_improvement, torch.tensor(0.75, device=dev), dim=-1, keepdim=True
            )
        patch_loss = (patch_xent * mask).mean()
        loss = loss + patch_loss

        metrics.push(
            loss=loss,
            loss_first=class_xent[:, 0].mean(),
            loss_last=class_xent[:, -1].mean(),
            xent_improvement=xent_improvement.mean(),
            patch_loss=patch_loss,
        )

        acc = (
            torch.argmax(class_logits, dim=-1) == labels_N.view(N, 1).expand(N, T)
        ).float()
        metrics.push(
            acc=acc.mean(),
            acc_first=acc[:, 0].mean(),
            acc_last=acc[:, -1].mean(),
        )

        return loss


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main_process = self.rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            dist.init_process_group(backend="nccl", device_id=self.device)

        log.info(f"using {DTYPE=}")
        random.seed(self.rank)
        np.random.seed(self.rank)
        torch.manual_seed(self.rank)
        torch.cuda.manual_seed(self.rank)

        metrics.enabled = self.is_main_process
        metrics.use_wandb = cfg.use_wandb
        metrics.context = "train_"

        dataset_path = Path(__file__).parent / "data"
        dataset_path.mkdir(exist_ok=True)

        self.train_loader = mnist_loader(
            train=True, batch_size=cfg.batch_size, device=self.device
        )
        self.transform = v2.Compose(
            [
                v2.RandomCrop(28, padding=2),
                v2.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0)),
            ]
        )

        self.model = FoveatedMnist(cfg.model).to(self.device)
        if cfg.do_compile:
            self.model = torch.compile(self.model)

        muon_params, scalar_params, embeds_params, output_params = (
            split_muon_adamw_params(self.model, verbose=True)
        )

        self.optims = []
        if len(muon_params) > 0:
            muon = Muon(
                muon_params,
                lr=self.cfg.muon_lr,
                momentum=0.9,
                weight_decay=self.cfg.wd,
            )
            self.optims.append(muon)
        adamw = torch.optim.AdamW(
            scalar_params + embeds_params + output_params,
            lr=self.cfg.adamw_lr,
            betas=[0.9, 0.99],
            weight_decay=self.cfg.wd,
        )
        self.optims.append(adamw)
        for opt in self.optims:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        self.step = 0

    def log_once(self, s, level=logging.INFO):
        if self.is_main_process:
            log.log(level, s)

    def schedule_lr(self):
        if (
            self.cfg.lr_cooldown_start is not None
            and self.step >= self.cfg.lr_cooldown_start
        ):
            frac = (self.step - self.cfg.lr_cooldown_start + 1) / (
                self.cfg.n_steps - self.cfg.lr_cooldown_start
            )
            relative_lr = 1.0 - (1 - self.cfg.lr_cooldown_ratio) * min(1.0, frac)
            for opt in self.optims:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * relative_lr
        else:
            relative_lr = 1.0
        metrics.push(relative_lr=relative_lr)

    def train_step(self):
        self.schedule_lr()

        images, labels = next(self.train_loader)
        if self.cfg.do_augment:
            images = self.transform(images.unsqueeze(-3))
            images = images.squeeze(-3)
            assert images.shape == (self.cfg.batch_size, 28, 28)

        loss = self.model(images, labels)
        loss.backward()
        for optim in self.optims:
            optim.step()
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()
        metrics.context = "valid_"
        for images, labels in mnist_loader(
            train=False, batch_size=self.cfg.batch_size, epochs=1, device=self.device
        ):
            self.model(images, labels)

        metrics.report()
        self.model.train()
        metrics.context = "train_"

    def run(self):
        with tqdm.tqdm(total=self.cfg.n_steps, desc="training") as progress_bar:
            if self.step == 0:
                self.log_once("running initial validation epoch")
                self.valid_epoch()
            else:
                progress_bar.update(self.step)
            while self.step < self.cfg.n_steps:
                self.train_step()

                if (
                    self.cfg.valid_every is not None
                    and (self.step + 1) % self.cfg.valid_every == 0
                ) or self.step + 1 == self.cfg.n_steps:
                    self.valid_epoch()

                self.step += 1
                progress_bar.update(1)


@parse_config
def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)
    try:
        trainer = Trainer(cfg)
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
