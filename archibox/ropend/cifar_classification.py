import logging
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from bnnp import Metrics, Muon, auto_split_muon_params, parse_config
from bnnp.nn import FusedLinear, MLPBlock, Output, RMSNorm
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from archibox.ropend.positional_embeddings import (
    AbsolutePE,
    AbsolutePEConfig,
    AxialRoPE,
    AxialRoPEConfig,
    FixedSinCosPE,
    FixedSinCosPEConfig,
    LieRE,
    LieREConfig,
    UniformRoPE,
    UniformRoPEConfig,
)

log = logging.getLogger(__name__)
metrics = Metrics(enabled=True, use_wandb=False, use_cuda_events=True)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32
COMPILE_MODE = "default"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patch_size: int = 4
    dim: int = 384
    mlp_dim: int = 768
    head_dim: int = 64
    depth: int = 6
    pos_emb: Literal["absolute", "fixed", "axial_rotary", "liere", "uniform_rotary"] = (
        "uniform_rotary"
    )
    ape_init_std: float = 0.5
    min_freq: float = 1.0
    max_freq: float = 100.0
    n_zero_freqs: int = 0
    direction_spacing: float | None = math.pi * (1 - math.sqrt(5))
    learnable_rope: bool = False
    sep_rope_heads: bool = True
    pooling: Literal["mean", "rotary", "attention"] = "mean"

    n_steps: int = 10_000
    valid_every: int = 500
    batch_size: int = 1000
    muon_lr: float = 0.06
    muon_mu: float = 0.95
    muon_wd: float = 0.01
    scalar_lr: float = 0.003
    embeds_lr: float = 0.003
    output_lr: float = 0.003
    adamw_mus: tuple[float, float] = (0.9, 0.95)
    adamw_wd: float = 0.01
    lr_cooldown_start: int | None = 7500
    lr_cooldown_ratio: float = 0.0

    do_augment: bool = True
    padding: int = 4
    label_smoothing: float = 0.1

    do_compile: bool = True
    debug: bool = False
    seed: int = 0
    log_param_info: bool = False


class EncoderSelfAttention(nn.Module):
    def __init__(
        self, dim: int, head_dim: int, rotary: AxialRoPE | LieRE | UniformRoPE | None
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.rotary = rotary

        self.in_norm = RMSNorm(dim)
        self.qkv_weight = nn.Parameter(torch.randn(3, dim, dim) / math.sqrt(dim) / 2)
        self.qk_norm = RMSNorm(head_dim)
        self.o_proj = FusedLinear(dim, dim, scale=True, zero_init=True)

    def forward(self, input_NHWD: torch.Tensor) -> torch.Tensor:
        N, H, W, D = input_NHWD.shape
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_NHWD)
        q_NHWhd, k_NHWhd, v_NHWhd = (
            (self.in_norm(input_NHWD) @ qkv_weight.t())
            .reshape(N, H, W, 3 * self.n_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q_NHWhd, k_NHWhd = self.qk_norm(q_NHWhd), self.qk_norm(k_NHWhd)
        if self.rotary is not None:
            q_NHWhd, k_NHWhd = self.rotary(q_NHWhd), self.rotary(k_NHWhd)
        x_NhLd = F.scaled_dot_product_attention(
            rearrange(q_NHWhd, "N H W h d -> N h (H W) d"),
            rearrange(k_NHWhd, "N H W h d -> N h (H W) d"),
            rearrange(v_NHWhd, "N H W h d -> N h (H W) d"),
        )
        x_NHWD = rearrange(x_NhLd, "N h (H W) d -> N H W (h d)", N=N, H=H, W=W)
        return input_NHWD + self.o_proj(x_NHWD)


class ViTClassifier(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        dim: int,
        mlp_dim: int,
        head_dim: int,
        depth: int,
        n_classes: int,
        pos_emb: AbsolutePE | AxialRoPE | FixedSinCosPE | UniformRoPE,
        pooling: str = "mean",
    ):
        super().__init__()
        assert image_size[0] % patch_size == 0
        assert image_size[1] % patch_size == 0
        self.patch_size = patch_size
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.n_heads = dim // head_dim
        self.head_dim = head_dim
        self.depth = depth
        self.n_classes = n_classes
        self.pooling = pooling
        self.patchify = FusedLinear(patch_size * patch_size * 3, dim)

        if pooling == "rotary":
            self.pool_rope = UniformRoPE(
                UniformRoPEConfig(
                    n_heads=1,
                    head_dim=dim,
                    min_freq=1.0,
                    max_freq=100.0,
                ),
                nh=image_size[0] // patch_size,
                nw=image_size[1] // patch_size,
            )
        elif pooling == "attention":
            self.pool_in_norm = RMSNorm(dim)
            self.pool_q = nn.Parameter(torch.randn(dim // head_dim, head_dim) / 2)
            self.pool_q._is_embed = True
            self.pool_weight_kv = nn.Parameter(torch.randn(2, dim, dim) / dim**0.5 / 2)
            self.pool_qk_norm = RMSNorm(head_dim)
            self.pool_rope = pos_emb
        else:
            assert pooling == "mean"

        self.pos_emb = pos_emb
        self.pos_embed_input = isinstance(self.pos_emb, (AbsolutePE, FixedSinCosPE))

        blocks = []
        rotary = None if self.pos_embed_input else self.pos_emb
        for _ in range(depth):
            blocks.append(EncoderSelfAttention(dim, head_dim, rotary))
            blocks.append(MLPBlock(dim, mlp_dim))
        self.blocks = nn.Sequential(*blocks)

        self.output = Output(dim, n_classes)

    def forward(self, images_NCHW: torch.Tensor) -> torch.Tensor:
        assert torch.is_floating_point(images_NCHW)
        input_NHWD = rearrange(
            images_NCHW,
            "N C (nh ph) (nw pw) -> N nh nw (ph pw C)",
            ph=self.patch_size,
            pw=self.patch_size,
            C=3,
        )
        input_NHWD = self.patchify(input_NHWD)
        if self.pos_embed_input:
            input_NHWD = self.pos_emb(input_NHWD)

        x_NHWD = self.blocks(input_NHWD)
        if self.pooling == "rotary":
            x_ND = self.pool_rope(x_NHWD.unsqueeze(-2)).squeeze(dim=-2).mean(dim=(1, 2))
        elif self.pooling == "attention":
            N, H, W, D = x_NHWD.shape
            weight_kv = self.pool_weight_kv.flatten(0, 1).type_as(x_NHWD)
            k_NHWhd, v_NHWhd = (
                (self.pool_in_norm(x_NHWD) @ weight_kv.t())
                .reshape(N, H, W, 2 * self.n_heads, self.head_dim)
                .chunk(2, dim=-2)
            )
            q_hd, k_NHWhd = self.pool_qk_norm(self.pool_q), self.pool_qk_norm(k_NHWhd)
            q_N1hd = q_hd.expand(N, 1, -1, -1).type_as(x_NHWD)
            if not self.pos_embed_input:
                k_NHWhd = self.pool_rope(k_NHWhd)
            x_Nh1d = F.scaled_dot_product_attention(
                rearrange(q_N1hd, "N 1 h d -> N h 1 d"),
                rearrange(k_NHWhd, "N H W h d -> N h (H W) d"),
                rearrange(v_NHWhd, "N H W h d -> N h (H W) d"),
            )
            x_ND = rearrange(x_Nh1d, "N h 1 d -> N (h d)")
        else:
            assert self.pooling == "mean"
            x_ND = x_NHWD.mean(dim=(1, 2))

        return self.output(x_ND)


def cifar_loader(
    train: bool,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    augmentation: v2.Transform | None = None,
    normalization: v2.Transform | None = None,
    limit_epochs: int | None = None,
):
    dataset = CIFAR10("archibox/cifar/data/", train=train)
    images_NHWC = dataset.data
    labels_N = torch.tensor(dataset.targets, device=device)

    images_NHWC = torch.tensor(images_NHWC, device=device, dtype=torch.uint8)
    images_NCHW = images_NHWC.movedim(3, 1)

    epoch = 0
    while limit_epochs is None or epoch < limit_epochs:
        n_batches = len(images_NCHW) // (batch_size * world_size)
        rng = torch.Generator(device)
        rng.manual_seed(epoch + 196613 * rank)
        idx = torch.randperm(len(images_NCHW), generator=rng, device=device)
        idx = idx[rank * batch_size * n_batches : (rank + 1) * batch_size * n_batches]

        shuffled_images_NCHW = images_NCHW[idx].clone()
        shuffled_labels_N = labels_N[idx].clone()

        if augmentation is not None:
            # pre-augment in batches
            for i in range(0, len(shuffled_images_NCHW), 1000):
                shuffled_images_NCHW[i : i + 1000] = augmentation(
                    shuffled_images_NCHW[i : i + 1000]
                )

            # shuffle again
            idx = torch.randperm(
                len(shuffled_images_NCHW), generator=rng, device=device
            )
            shuffled_images_NCHW = shuffled_images_NCHW[idx].contiguous()
            shuffled_labels_N = shuffled_labels_N[idx].contiguous()

        if normalization is not None:
            shuffled_images_NCHW = normalization(shuffled_images_NCHW)

        for i in range(n_batches):
            yield (
                shuffled_images_NCHW[batch_size * i : batch_size * (i + 1)],
                shuffled_labels_N[batch_size * i : batch_size * (i + 1)],
            )
        epoch += 1


@parse_config
def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)

    if not cfg.debug:
        metrics.use_wandb = True
        wandb.init(
            entity="archibox",
            project="ropend-cifar-class",
            group="bc3",
            dir=Path(__file__).parent / "runs",
            config=cfg.model_dump(),
        )
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    nh, nw = 32 // cfg.patch_size, 32 // cfg.patch_size
    if cfg.pos_emb == "absolute":
        pos_emb = AbsolutePE(
            AbsolutePEConfig(dim=cfg.dim, init_std=cfg.ape_init_std), nh, nw
        )
    elif cfg.pos_emb == "fixed":
        pos_emb = FixedSinCosPE(
            FixedSinCosPEConfig(
                dim=cfg.dim, min_freq=cfg.min_freq, max_freq=cfg.max_freq
            ),
            nh,
            nw,
        )
    elif cfg.pos_emb == "axial_rotary":
        pos_emb = AxialRoPE(
            AxialRoPEConfig(
                head_dim=cfg.head_dim,
                min_freq=cfg.min_freq,
                max_freq=cfg.max_freq,
                n_zero_freqs=cfg.n_zero_freqs,
            ),
            nh,
            nw,
        )
    elif cfg.pos_emb == "liere":
        pos_emb = LieRE(
            LieREConfig(
                # n_heads=cfg.dim // cfg.head_dim if cfg.sep_rope_heads else 1,
                head_dim=cfg.head_dim,
            ),
            nh,
            nw,
        )
    else:
        assert cfg.pos_emb == "uniform_rotary"
        pos_emb = UniformRoPE(
            UniformRoPEConfig(
                n_heads=cfg.dim // cfg.head_dim if cfg.sep_rope_heads else 1,
                head_dim=cfg.head_dim,
                min_freq=cfg.min_freq,
                max_freq=cfg.max_freq,
                n_zero_freqs=cfg.n_zero_freqs,
                direction_spacing=cfg.direction_spacing,
                learnable=cfg.learnable_rope,
            ),
            nh,
            nw,
        )
    raw_model = ViTClassifier(
        image_size=(32, 32),
        patch_size=cfg.patch_size,
        dim=cfg.dim,
        mlp_dim=cfg.mlp_dim,
        head_dim=cfg.head_dim,
        depth=cfg.depth,
        n_classes=10,
        pos_emb=pos_emb,
        pooling=cfg.pooling,
    )
    raw_model.cuda()
    model = torch.compile(raw_model, mode=COMPILE_MODE) if cfg.do_compile else raw_model
    muon_params, scalar_params, embeds_params, output_params = auto_split_muon_params(
        raw_model, log_level=logging.INFO if cfg.log_param_info else logging.DEBUG
    )
    adamw_params = [
        dict(params=scalar_params, lr=cfg.scalar_lr),
        dict(params=embeds_params, lr=cfg.embeds_lr),
        dict(params=output_params, lr=cfg.output_lr),
    ]
    muon = Muon(
        muon_params, lr=cfg.muon_lr, momentum=cfg.muon_mu, weight_decay=cfg.muon_wd
    )
    adamw = torch.optim.AdamW(
        adamw_params, betas=cfg.adamw_mus, weight_decay=cfg.adamw_wd
    )
    for opt in (muon, adamw):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    if cfg.do_augment:
        augmentation = v2.Compose(
            [
                v2.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.025
                ),
                v2.RandomCrop(32, padding=cfg.padding, padding_mode="edge"),
                v2.RandomHorizontalFlip(),
            ]
        )
    normalization = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    train_loader = iter(
        cifar_loader(
            train=True,
            batch_size=cfg.batch_size,
            device="cuda",
            augmentation=augmentation if cfg.do_augment else None,
            normalization=normalization,
        )
    )

    metrics.context = "train_"
    for step in tqdm.trange(cfg.n_steps):
        if (
            cfg.lr_cooldown_start is not None
            and cfg.lr_cooldown_start <= step
            and cfg.lr_cooldown_start < cfg.n_steps
        ):
            frac = (step - cfg.lr_cooldown_start + 1) / (
                cfg.n_steps - cfg.lr_cooldown_start
            )
            relative_lr = 1.0 - (1.0 - cfg.lr_cooldown_ratio) * min(1.0, frac)
            for opt in (muon, adamw):
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * relative_lr
        else:
            relative_lr = 1.0
        metrics.push(relative_lr=relative_lr)

        metrics.tick("load_batch")
        images_NCHW, labels_N = next(train_loader)

        metrics.tick("forward")
        logits_ND = model(images_NCHW.type(DTYPE))
        assert logits_ND.dtype == DTYPE
        loss = F.cross_entropy(
            logits_ND.float(), labels_N, label_smoothing=cfg.label_smoothing
        )
        acc = (torch.argmax(logits_ND, dim=-1) == labels_N).float().mean()
        metrics.push(loss=loss, acc=acc)

        metrics.tick("backward")
        loss.backward()

        metrics.tick("optim")
        for optim in (muon, adamw):
            optim.step()
        for optim in (muon, adamw):
            optim.zero_grad(set_to_none=True)

        metrics.tick(None)

        if (step + 1) % cfg.valid_every == 0:
            model.eval()
            metrics.context = "valid_"
            with torch.no_grad():
                for images_NCHW, labels_N in cifar_loader(
                    train=False,
                    batch_size=cfg.batch_size,
                    limit_epochs=1,
                    augmentation=None,
                    normalization=normalization,
                    device="cuda",
                ):
                    logits_ND = model(images_NCHW.type(DTYPE))
                    loss = F.cross_entropy(
                        logits_ND.float(), labels_N, label_smoothing=cfg.label_smoothing
                    )
                    nll = F.cross_entropy(logits_ND.float(), labels_N)
                    acc = (torch.argmax(logits_ND, dim=-1) == labels_N).float().mean()
                    metrics.push(loss=loss, nll=nll, acc=acc)

            metrics.report()

            model.train()
            metrics.context = "train_"


if __name__ == "__main__":
    main()
