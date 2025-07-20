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
from pathlib import Path

import matplotlib
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from bnnp import Metrics, parse_config
from bnnp.nn import Output
from pydantic import BaseModel, Field
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

from archibox.foveation.patching import (
    PatchDecoder,
    build_patch_locs,
    extract_patches,
    visualize_patches,
)
from archibox.trainer import Trainer, TrainerConfig
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
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            run_dir=None,
            n_steps=1_000,
            valid_every=250,
            save_every=None,
            micro_batch_size=2048,
            train_loader_workers=0,
            valid_loader_workers=0,
            muon_lr=0.01,
            muon_wd=0.01,
            scalar_lr=0.0005,
            embeds_lr=0.0005,
            output_lr=0.0005,
            adamw_wd=0.01,
            adamw_mu1=0.9,
            adamw_mu2=0.99,
        )
    )


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
        output_NTD = self.net(patches_NThw.unsqueeze(-3), pos_NT2, sizes_NT)
        class_logits = self.output_head(output_NTD).float()

        # Compute patch logits
        if self.cfg.detach_patch_head:
            patch_logits = self.patch_head(output_NTD[:, -1].detach())
        else:
            patch_logits = self.patch_head(output_NTD[:, -1])

        return class_logits, patch_logits

    def generate(self, images_NHW, n_uniform: int = 0):
        assert torch.is_floating_point(images_NHW)
        N, H, W = images_NHW.shape
        L = len(self.patch_locs)
        T = self.cfg.seq_len

        images_NHW = images_NHW.type(DTYPE)
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

    def forward(self, batch):
        images_NHW, labels_N = batch

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


class FoveatedMnistTrainer(Trainer):
    def on_validation_end(self):
        self.metrics.tick("visualization")
        images, labels = next(iter(self.valid_loader))
        images = images[:3]
        labels = labels[:3]
        class_logits, _, patch_idx = self.model.generate(images)
        for i in range(3):
            frames = visualize_patches(
                image_CHW=images[i].unsqueeze(0),
                label=labels[i].item(),
                class_logits=class_logits[i],
                patch_locs=self.model.patch_locs,
                patch_idx=patch_idx[i],
                P=self.model.cfg.patch_size,
            )
            name = f"data/step_{self.step + 1 if self.step > 0 else 0}_sample_{i + 1}"
            save_frames_to_gif(frames, Path(__file__).parent / f"{name}.gif", fps=5)
            save_frames_to_mp4(frames, Path(__file__).parent / f"{name}.mp4", fps=5)
        self.metrics.tick(None)

    def on_epoch_end(self):
        pass


@parse_config
def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)

    model = FoveatedMnist(cfg.model)

    mnist_train = MNIST(Path(__file__).parent / "data", train=True, download=True)
    mnist_valid = MNIST(Path(__file__).parent / "data", train=False, download=True)

    # For MNIST, transfer entire dataset to GPU beforehand for speed.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    train_dataset = TensorDataset(
        mnist_train.data.to(device, DTYPE) / 255.0 * 2 - 1,
        mnist_train.targets.to(device),
    )
    valid_dataset = TensorDataset(
        mnist_valid.data.to(device, DTYPE) / 255.0 * 2 - 1,
        mnist_valid.targets.to(device),
    )

    try:
        trainer = FoveatedMnistTrainer(
            cfg.trainer, model, train_dataset, valid_dataset, metrics
        )
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
