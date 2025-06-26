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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from bnnp import Metrics, Muon, auto_split_muon_params, parse_config
from bnnp.nn import FusedLinear, Output
from einops import rearrange
from pydantic import BaseModel, Field
from torchvision.transforms import v2

from archibox.foveation.multirotary import MultiRotaryDecoder
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.utils import save_frames_to_gif, save_frames_to_mp4

matplotlib.use("agg")
sns.set_theme()
log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32
metrics = Metrics(enabled=False, use_wandb=False)


class ModelConfig(BaseModel):
    nlevels: int = 3
    block_size: int = 4  # number of patches to sample at a time
    patch_size: int = 8  # side length of resized patches, in pixels
    seq_len: int = 16
    overlap_patches: bool = True
    detach_patch_head: bool = False
    p_uniform: float = 0.1

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
    batch_size: int = 2048
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

        # Include both absolute and relative positional embedding
        self.pos_embed = FourierEmbed(3, nfreqs, min_freq=0.1, max_freq=100.0)
        self.emb_proj = FusedLinear(nfreqs * 2, dim)

        self.decoder = MultiRotaryDecoder(
            dim,
            mlp_dim,
            head_dim=64,
            pos_dim=3,
            depth=depth,
            seq_len=seq_len,
            use_rope=True,
            min_freq=0.1,
            max_freq=100.0,
            frozen_rope=True,
        )

    def forward(self, patches_NTHWC, pos_NT2, sizes_NT):
        """
        Args:
            patches_NTHWC: patches (should be already resized) extracted from images
            pos_NT2: coordinates of patch centers in [-1, 1].
                -1 is bottom/left, +1 is top/right
            sizes_NT: sizes of patches in [0, 1]
                0 is size 0, 1 is full image
        """
        assert torch.is_floating_point(patches_NTHWC)
        N, T, _, _, _ = patches_NTHWC.shape

        patches_NTD = rearrange(patches_NTHWC, "N T H W C -> N T (H W C)")
        x_NTD = self.patchify(patches_NTD)

        pos_NT3 = torch.cat([pos_NT2, sizes_NT.unsqueeze(-1)], dim=-1)

        pos_emb = self.pos_embed(pos_NT3).type_as(x_NTD)
        emb_NTD = self.emb_proj(pos_emb)
        return self.decoder(x_NTD + emb_NTD, pos_NT3)


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


def extract_patches(
    images_NCHW: torch.Tensor,
    y0_NT: torch.Tensor,
    x0_NT: torch.Tensor,
    y1_NT: torch.Tensor,
    x1_NT: torch.Tensor,
    P: int,
):
    """Extracts patches from batch of images using area-style intepolation.

    T: number of patches per image
    P: resized patch side length, in pixels
    """
    assert torch.is_floating_point(images_NCHW)
    N, C, H, W = images_NCHW.shape
    _, T = y0_NT.shape
    assert y0_NT.shape == y1_NT.shape == x0_NT.shape == x1_NT.shape == (N, T)

    offsets = torch.linspace(0, 1, P + 1, device=images_NCHW.device)
    ys_NTP = y0_NT[..., None] + (y1_NT - y0_NT)[..., None] * offsets
    xs_NTP = x0_NT[..., None] + (x1_NT - x0_NT)[..., None] * offsets
    ys_NTPP = ys_NTP.view(N, T, P + 1, 1).expand(N, T, P + 1, P + 1)
    xs_NTPP = xs_NTP.view(N, T, 1, P + 1).expand(N, T, P + 1, P + 1)
    grid_xy_NTPP2 = torch.stack([xs_NTPP, ys_NTPP], dim=-1)

    integral = torch.zeros((N, C, H + 1, W + 1), device=images_NCHW.device)
    integral[:, :, 1:, 1:] = images_NCHW.cumsum(dim=-1).cumsum(dim=-2)

    integral_samples = F.grid_sample(
        einops.repeat(integral, "N C H W -> (N T) C H W", T=T),
        grid_xy_NTPP2.reshape(N * T, P + 1, P + 1, 2).float(),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).reshape(N, T, C, P + 1, P + 1)

    i00 = integral_samples[..., :-1, :-1]
    i01 = integral_samples[..., :-1, 1:]
    i10 = integral_samples[..., 1:, :-1]
    i11 = integral_samples[..., 1:, 1:]

    # Box-sum via 4-corner trick, then divide by pixel area
    sums = i11 - i01 - i10 + i00
    area = (y1_NT - y0_NT) * H / 2 / P * (x1_NT - x0_NT) * W / 2 / P
    patches = sums / area.view(N, T, 1, 1, 1)
    return patches.type_as(images_NCHW)


def sample_topk(logits: torch.Tensor, k: int):
    assert 1 <= k <= logits.size(-1)
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u))
    scores = logits + gumbel
    _, idx = torch.topk(scores, k, dim=-1, sorted=True)
    return idx


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
            seq_len=cfg.block_size,
        )

        self.patch_locs = nn.Buffer(
            build_patch_locs(nlevels=cfg.nlevels, overlap=cfg.overlap_patches)
        )
        assert -1.0 <= self.patch_locs.amin()
        assert self.patch_locs.amax() <= 1.0

        self.output_head = Output(cfg.dim, 10)
        self.patch_head = Output(cfg.dim, len(self.patch_locs), pad_to=64)
        self.initial_logits = nn.Parameter(torch.zeros(len(self.patch_locs)))

    def predict(self, images_NHW, patch_idx):
        N, H, W = images_NHW.shape
        _, T = patch_idx.shape
        P = self.cfg.patch_size
        assert patch_idx.size(0) == N
        assert P <= min(W, H)

        y0, x0, y1, x1 = self.patch_locs[patch_idx].unbind(-1)
        patches_NThw = extract_patches(
            images_NHW.unsqueeze(-3), y0, x0, y1, x1, P
        ).squeeze(-3)

        pos_NT2 = torch.stack([(y0 + y1) / 2, (x0 + x1) / 2], dim=-1)
        sizes_NT = (y1 - y0) / 2

        # Compute class logits
        output_NTD = self.net(patches_NThw.unsqueeze(-1), pos_NT2, sizes_NT)
        class_logits = self.output_head(output_NTD).float()

        # Compute patch logits
        if self.cfg.detach_patch_head:
            patch_logits = self.patch_head(output_NTD[:, -1].detach())
        else:
            patch_logits = self.patch_head(output_NTD[:, -1])

        return class_logits, patch_logits

    def generate(self, images_NHW, n_uniform: int = 0):
        assert images_NHW.dtype == torch.uint8
        N, H, W = images_NHW.shape
        L = len(self.patch_locs)
        T = self.cfg.seq_len

        images_NHW = images_NHW.type(DTYPE) / 255 * 2 - 1
        all_class_logits = []
        all_patch_logits = []
        all_patch_idx = []
        for b in range(T // self.cfg.block_size):
            k = self.cfg.block_size
            if b == 0:
                patch_logits = self.initial_logits.expand(N, L)
                patch_idx = sample_topk(patch_logits, k)
            else:
                patch_idx = sample_topk(patch_logits, k)
            patch_idx[:n_uniform] = torch.randint(
                0, L, (n_uniform, k), device=images_NHW.device
            )

            all_patch_logits.append(patch_logits.view(N, 1, L).expand(N, k, L))
            all_patch_idx.append(patch_idx)

            class_logits, patch_logits = self.predict(images_NHW, patch_idx)
            all_class_logits.append(class_logits)

        class_logits = torch.cat(all_class_logits, dim=1)
        patch_logits = torch.cat(all_patch_logits, dim=1)
        patch_idx = torch.cat(all_patch_idx, dim=1)
        return class_logits, patch_logits, patch_idx

    def forward(self, images_NHW, labels_N):
        N, H, W = images_NHW.shape
        T = self.cfg.seq_len
        n_uniform = round(N * self.cfg.p_uniform)
        dev = images_NHW.device

        class_logits, patch_logits, patch_idx = self.generate(images_NHW, n_uniform)
        assert class_logits.shape == (N, T, 10)
        assert patch_logits.shape == (N, T, len(self.patch_locs))
        assert patch_idx.shape == (N, T)

        class_xent = F.cross_entropy(
            class_logits.flatten(0, -2),
            labels_N.view(N, 1).expand(N, T).flatten(),
            reduction="none",
        ).reshape(N, T)
        loss = class_xent.mean()

        patch_xent = F.cross_entropy(
            patch_logits.flatten(0, -2), patch_idx.flatten(), reduction="none"
        ).reshape(N, T)
        with torch.no_grad():
            # Using mean for first so that logged xent_improvement is reasonable.
            # Shouldn't impact learning in general.
            previous_xent = torch.cat(
                [class_xent[:, 0].mean().expand(N, 1), class_xent[:, :-1]], dim=1
            )
            xent_improvement = -(class_xent - previous_xent).float()
            q90 = torch.quantile(
                xent_improvement, torch.tensor(0.9, device=dev), dim=0, keepdim=True
            )
            advantage = torch.where(xent_improvement > q90, 1.0, 0.0)
        patch_loss = (patch_xent * advantage).mean()
        loss = loss + patch_loss

        metrics.push(
            loss=loss,
            patch_loss=patch_loss,
            xent=class_xent.mean(),
            xent_first=class_xent[:, 0].mean(),
            xent_last=class_xent[:, -1].mean(),
            xent_uniform=class_xent[:n_uniform].mean(),
            xent_nonunif=class_xent[n_uniform:].mean(),
            xent_nonunif_last=class_xent[n_uniform:, -1].mean(),
            xent_improvement=xent_improvement.mean(),
        )

        acc = (
            torch.argmax(class_logits, dim=-1) == labels_N.view(N, 1).expand(N, T)
        ).float()
        metrics.push(
            acc=acc.mean(),
            acc_first=acc[:, 0].mean(),
            acc_last=acc[:, -1].mean(),
            acc_uniform=acc[:n_uniform].mean(),
            acc_nonunif=acc[n_uniform:].mean(),
            acc_nonunif_last=acc[n_uniform:, -1].mean(),
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
            auto_split_muon_params(self.model, log_level=logging.INFO)
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

        # Gather validation metrics: loss, accuracy, etc.
        for images, labels in mnist_loader(
            train=False, batch_size=self.cfg.batch_size, epochs=1, device=self.device
        ):
            self.model(images, labels)

        # Save video examples
        images, labels = next(
            iter(mnist_loader(train=False, batch_size=3, epochs=1, device=self.device))
        )
        class_logits, _, patch_idx = self.model.generate(images)
        probs = F.softmax(class_logits, dim=-1)
        for i in range(3):
            img = images[i].cpu()

            frames = []
            fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
            fig.set_size_inches(10, 10)
            axes[0, 0].set_title("Full image")
            axes[0, 0].imshow(img, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].set_title("Model views")
            axes[0, 1].set_xlim(-1, 1)
            axes[0, 1].set_ylim(1, -1)
            axes[0, 1].axis("off")

            for t in range(self.model.cfg.seq_len):
                y0, x0, y1, x1 = self.model.patch_locs[patch_idx[i, t]].cpu()
                P = self.model.cfg.patch_size
                patch = extract_patches(
                    img.view(1, 1, 28, 28).float() / 255.0,
                    y0.view(1, 1),
                    x0.view(1, 1),
                    y1.view(1, 1),
                    x1.view(1, 1),
                    P,
                )
                assert patch.shape == (1, 1, 1, P, P)
                patch = (patch.view(P, P).clamp(0, 1) * 255.0).type(torch.uint8)
                patch = patch.numpy()
                # half_pixel = (y1 - y0) / P / 2
                axes[0, 1].imshow(
                    patch, origin="upper", extent=(x0, x1, y1, y0), cmap="gray"
                )

                axes[1, 0].clear()
                axes[1, 0].set_title("Class distribution")
                axes[1, 0].barh(np.arange(10), probs[i, t].cpu().numpy())
                axes[1, 0].set_xlim(0, 1.05)
                axes[1, 0].set_ylim(10, -1)
                axes[1, 0].set_yticks(np.arange(10))

                axes[1, 1].clear()
                axes[1, 1].set_title(f"P(label == {int(labels[i])})")
                axes[1, 1].set_xlim(0, self.model.cfg.seq_len)
                axes[1, 1].set_ylim(-0.05, 1.05)
                axes[1, 1].plot(probs[i, : t + 1, int(labels[i])].float().cpu().numpy())

                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3])
            plt.close(fig)

            save_frames_to_mp4(
                frames,
                Path(__file__).parent
                / f"data/step_{self.step + 1 if self.step > 0 else 0}_sample_{i + 1}.mp4",
                fps=5,
            )
            save_frames_to_gif(
                frames,
                Path(__file__).parent
                / f"data/step_{self.step + 1 if self.step > 0 else 0}_sample_{i + 1}.gif",
                fps=5,
            )

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
