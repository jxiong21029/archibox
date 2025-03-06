import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from sandbox.cifar.architecture import VisionTransformer
from sandbox.components import FusedLinear
from sandbox.muon import Muon

log = logging.getLogger(__name__)
is_main_process = True


class ViTClassifier(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.vit = VisionTransformer(32, 32, patch_size=8, dim=dim, depth=depth)
        self.class_head = FusedLinear(dim, 10)
        self.class_head.weight.data.zero_()
        self.class_head._ortho = False

    def setup_optimizers(
        self, embeds_lr, scalars_lr, heads_lr, adamw_kwargs, muon_lr, muon_kwargs
    ):
        adamw_embeds = []
        adamw_scalars = []
        adamw_heads = []
        muon_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                # log.info(f"{name} {p.shape} requires_grad=False, skipping...")
                pass
            elif p.ndim <= 1:
                # log.info(f"{name} {p.shape} assigned to AdamW (scalar)")
                adamw_scalars.append(p)
            elif name.endswith("class_head.weight"):
                # log.info(f"{name} {p.shape} assigned to AdamW (output head)")
                adamw_heads.append(p)
            elif name.endswith("vit.pos_emb"):
                adamw_embeds.append(p)
            elif not hasattr(p, "_ortho") or p._ortho:
                # log.info(f"{name} {p.shape} assigned to Muon")
                muon_params.append(p)
            else:
                log.warning(f"{name} {p.shape} unasssigned (?)")

        adamw_p = sum(p.numel() for p in adamw_embeds + adamw_scalars + adamw_heads)
        adamw_n = len(adamw_embeds + adamw_scalars + adamw_heads)
        muon_p = sum(p.numel() for p in muon_params)
        muon_n = len(muon_params)
        log.info(f"AdamW: {adamw_p:,} params across {adamw_n:,} tensors")
        log.info(f"Muon: {muon_p:,} params across {muon_n:,} tensors")

        adamw = torch.optim.AdamW(
            [
                dict(params=adamw_embeds, lr=embeds_lr),
                dict(params=adamw_scalars, lr=scalars_lr),
                dict(params=adamw_heads, lr=heads_lr),
            ],
            **adamw_kwargs,
        )
        muon = Muon(muon_params, lr=muon_lr, **muon_kwargs)
        return [adamw, muon]

    def forward(self, images: Tensor, labels: Tensor):
        N, C, H, W = images.shape
        assert torch.is_floating_point(images)
        assert C == 3
        assert labels.shape == (N,)

        x_ND = self.vit(images)
        logits = self.class_head(x_ND).float()
        loss = F.cross_entropy(logits, labels)
        acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        return loss, acc


@dataclass
class Config:
    n_steps: int = 400
    batch_size: int = 1024
    valid_size: int = 0.01
    valid_every: int = 25

    dim: int = 256
    depth: int = 8

    embeds_lr: float = 0.01
    scalars_lr: float = 0.01
    heads_lr: float = 0.05
    adamw_betas: list[float] = field(default_factory=lambda: [0.8, 0.95])
    adamw_wd: float = 0.0
    muon_lr: float = 0.2
    muon_momentum: float = 0.8
    muon_wd: float = 0.0


def main():
    cfg = Config()
    logging.basicConfig(level=logging.INFO)

    device = "cuda:0"

    dataset = CIFAR10(
        Path(__file__).parent / "data",
        train=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.bfloat16, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    )
    train_dataset, valid_dataset = random_split(
        dataset, [1 - cfg.valid_size, cfg.valid_size]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size)

    model = ViTClassifier(dim=cfg.dim, depth=cfg.depth).to(device=device)
    raw_model = model
    model = torch.compile(model)

    transform = v2.Compose(
        [v2.RandomHorizontalFlip(), v2.RandomCrop(size=32, padding=2)]
    )

    optims = raw_model.setup_optimizers(
        embeds_lr=cfg.embeds_lr,
        scalars_lr=cfg.scalars_lr,
        heads_lr=cfg.heads_lr,
        adamw_kwargs=dict(betas=cfg.adamw_betas, weight_decay=cfg.adamw_wd),
        muon_lr=cfg.muon_lr,
        muon_kwargs=dict(momentum=cfg.muon_momentum, weight_decay=cfg.muon_wd),
    )

    epoch = 0
    log.info(f"starting {epoch=}")
    train_loader_iter = iter(train_loader)
    train_loss = 0
    train_acc = 0
    for step in tqdm.trange(0, cfg.n_steps + 1):
        is_last_step = step == cfg.n_steps

        if is_last_step or (cfg.valid_every > 0 and step % cfg.valid_every == 0):
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_acc = 0
                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    loss, acc = model(images, labels)
                    valid_loss += loss / len(valid_loader)
                    valid_acc += acc / len(valid_loader)
                if step == 0:
                    log.info(f"{valid_loss=:.4f}, {valid_acc=:.2%}")
                else:
                    log.info(
                        f"train_loss={train_loss / cfg.valid_every:.4f}, "
                        f"train_acc={train_acc / cfg.valid_every:.2%} || "
                        f"{valid_loss=:.4f}, {valid_acc=:.2%}"
                    )
                train_loss = 0
                train_acc = 0
            model.train()

        if not is_last_step:
            try:
                images, labels = next(train_loader_iter)
            except StopIteration:
                epoch += 1
                log.info(f"starting {epoch=}")
                train_loader_iter = iter(train_loader)
                images, labels = next(train_loader_iter)

            images = images.to(device)
            labels = labels.to(device)

            images = transform(images)
            loss, acc = model(images, labels)
            loss.backward()
            for opt in optims:
                opt.step()
            for opt in optims:
                opt.zero_grad(set_to_none=True)

            train_loss += loss.detach()
            train_acc += acc.detach()


if __name__ == "__main__":
    main()
