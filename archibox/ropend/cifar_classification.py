import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from bnnp import Metrics, Muon, auto_split_muon_params, parse_config
from bnnp.nn import FusedLinear, MLPBlock, Output, RMSNorm
from einops import rearrange
from pydantic import BaseModel, ConfigDict, Field
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from archibox.ropend.positional_embeddings import (
    AbsolutePE,
    AbsolutePEConfig,
    AxialRoPE,
    AxialRoPEConfig,
    FixedSinCosPE,
    FixedSinCosPEConfig,
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
    pos_embedding: (
        AbsolutePEConfig | AxialRoPEConfig | FixedSinCosPEConfig | UniformRoPEConfig
    ) = Field(
        default_factory=lambda: UniformRoPEConfig(
            head_dim=64,
            min_freq=1.0,
            max_freq=100.0,
            direction_spacing=math.pi * (1 - math.sqrt(5)),
        ),
        discriminator="variant",
    )
    # rotary_pool: bool = False

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


# class Rotary2d(nn.Module):
#     def __init__(
#         self,
#         input_size: tuple[int, int],
#         head_dim: int,
#         min_freq: int,
#         max_freq: int,
#         direction_spacing: float = math.pi * (1 - math.sqrt(5)),
#     ):
#         super().__init__()
#         self.input_size = input_size
#         self.head_dim = head_dim
#         self.min_freq = min_freq
#         self.max_freq = max_freq
#         self.direction_spacing = direction_spacing

#         n_freqs = head_dim // 2
#         freqs_F = min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs)
#         phi_F = torch.arange(n_freqs) * direction_spacing
#         u_F2 = torch.stack((torch.cos(phi_F), torch.sin(phi_F)), dim=-1)

#         H, W = input_size
#         y = torch.linspace(-1, 1, H).reshape(H, 1).expand(H, W)
#         x = torch.linspace(-1, 1, W).reshape(1, W).expand(H, W)
#         positions_HW12 = torch.stack((y, x), dim=-1).reshape(H, W, 1, 2)
#         theta_HWF = (u_F2 * freqs_F.reshape(n_freqs, 1) * positions_HW12).mean(dim=-1)
#         self.register_buffer("cos_HW1F", torch.cos(theta_HWF).reshape(H, W, 1, n_freqs))
#         self.register_buffer("sin_HW1F", torch.sin(theta_HWF).reshape(H, W, 1, n_freqs))

#     def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
#         x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
#         x_out_NHWhF = x_NHWhF * self.cos_HW1F - y_NHWhF * self.sin_HW1F
#         y_out_NHWhF = x_NHWhF * self.sin_HW1F + y_NHWhF * self.cos_HW1F
#         output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
#         return output_NHWhd.type_as(input_NHWhd)


class EncoderSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, rotary: AxialRoPE | UniformRoPE | None):
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
        pos_emb: AbsolutePEConfig
        | AxialRoPEConfig
        | FixedSinCosPEConfig
        | UniformRoPEConfig,
        # rotary_pool: bool = False,
    ):
        super().__init__()
        assert image_size[0] % patch_size == 0
        assert image_size[1] % patch_size == 0
        nh = image_size[0] // patch_size
        nw = image_size[1] // patch_size
        self.patch_size = patch_size
        self.patchify = FusedLinear(patch_size * patch_size * 3, dim)

        if isinstance(pos_emb, AbsolutePEConfig):
            self.pos_emb = AbsolutePE(pos_emb, nh, nw)
            self.input_pe = True
        elif isinstance(pos_emb, AxialRoPEConfig):
            self.pos_emb = AxialRoPE(pos_emb, nh, nw)
            self.input_pe = False
        elif isinstance(pos_emb, FixedSinCosPEConfig):
            self.pos_emb = FixedSinCosPE(pos_emb, nh, nw)
            self.input_pe = True
        else:
            assert isinstance(pos_emb, UniformRoPEConfig)
            self.pos_emb = UniformRoPEConfig(pos_emb, nh, nw)
            self.input_pe = False

        # if rotary_pool:
        #     self.pool_rotation = Rotary2d(
        #         (nh, nw),
        #         dim,
        #         min_freq=1.0,
        #         max_freq=100.0,
        #         direction_spacing=math.pi * (math.sqrt(5) - 1),
        #     )
        # else:
        #     self.pool_rotation = None

        blocks = []
        rotary = None if self.input_pe else self.pos_emb
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
        if self.input_pe:
            input_NHWD = self.pos_emb(input_NHWD)

        x_NHWD = self.blocks(input_NHWD)
        if self.pool_rope is None:
            x_ND = x_NHWD.mean(dim=(1, 2))
        else:
            x_ND = (
                self.pool_rotation(x_NHWD.unsqueeze(-2))
                .squeeze(dim=-2)
                .mean(dim=(1, 2))
            )
        return self.output(x_ND)


def cifar_loader(
    train: bool,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    limit_epochs: int | None = None,
):
    dataset = CIFAR10("archibox/cifar/data/", train=train)
    images_NHWC = dataset.data
    labels_N = torch.tensor(dataset.targets, device=device)

    # mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device)
    # std = torch.tensor((0.247, 0.243, 0.261), device=device)
    # images_NHWC = (images_NHWC - mean) / std
    images_NHWC = torch.tensor(images_NHWC, device=device, dtype=torch.uint8)
    images_NCHW = images_NHWC.movedim(3, 1)

    epoch = 0
    while limit_epochs is None or epoch < limit_epochs:
        n_batches = len(images_NCHW) // (batch_size * world_size)
        rng = torch.Generator(device)
        rng.manual_seed(epoch + 196613 * rank)
        idx = torch.randperm(len(images_NCHW), generator=rng, device=device)
        idx = idx[rank * batch_size * n_batches : (rank + 1) * batch_size * n_batches]

        shuffled_images_NCHW = images_NCHW[idx].contiguous()
        shuffled_labels_N = labels_N[idx].contiguous()

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
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    raw_model = ViTClassifier(
        image_size=(32, 32),
        patch_size=cfg.patch_size,
        dim=cfg.dim,
        mlp_dim=cfg.mlp_dim,
        head_dim=cfg.head_dim,
        depth=cfg.depth,
        n_classes=10,
        pos_emb=cfg.pos_embedding,
        # rotary_pool=cfg.rotary_pool,
    )
    raw_model.cuda()
    model = torch.compile(raw_model, mode=COMPILE_MODE) if cfg.do_compile else raw_model
    muon_params, scalar_params, embeds_params, output_params = auto_split_muon_params(
        raw_model, log_level=logging.DEBUG
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

    train_loader = iter(
        cifar_loader(train=True, batch_size=cfg.batch_size, device="cuda")
    )
    if cfg.do_augment:
        augment = v2.Compose(
            [
                v2.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.025
                ),
                v2.RandomCrop(32, padding=cfg.padding, padding_mode="edge"),
                v2.RandomHorizontalFlip(),
            ]
        )
    normalize = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
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
        if cfg.do_augment:
            images_NCHW = augment(images_NCHW)
        images_NCHW = normalize(images_NCHW).type(DTYPE)

        metrics.tick("forward")
        logits_ND = model(images_NCHW)
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
                    device="cuda",
                ):
                    images_NCHW = normalize(images_NCHW).type(DTYPE)
                    logits_ND = model(images_NCHW)
                    loss = F.cross_entropy(
                        logits_ND.float(), labels_N, label_smoothing=cfg.label_smoothing
                    )
                    acc = (torch.argmax(logits_ND, dim=-1) == labels_N).float().mean()
                    metrics.push(loss=loss, acc=acc)

            metrics.report()

            model.train()
            metrics.context = "train_"


if __name__ == "__main__":
    main()
