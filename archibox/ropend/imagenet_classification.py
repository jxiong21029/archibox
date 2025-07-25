with open(__file__, "r") as f:
    CODE = f.read()

import logging
import math
import os
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from bnnp import Metrics, Muon, auto_split_muon_params, parse_config
from bnnp.nn import FusedLinear, MLPBlock, Output, RMSNorm
from einops import rearrange
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from pydantic import BaseModel, ConfigDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import v2

from archibox.ropend.dali_imagenet_loaders import (
    imagenet_train_pipeline,
    imagenet_valid_pipeline,
)
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
metrics = Metrics(enabled=False, use_wandb=False, use_cuda_events=True)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
    COMPILE_MODE = "max-autotune"
else:
    DTYPE = torch.float32
    COMPILE_MODE = "default"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patch_size: int = 16
    dim: int = 768
    mlp_dim: int = 3072
    head_dim: int = 64
    depth: int = 12
    pos_emb: Literal["absolute", "fixed", "axial_rotary", "uniform_rotary"] = (
        "uniform_rotary"
    )
    ape_init_std: float = 0.5
    min_freq: float = 0.2
    max_freq: float = 20.0
    n_zero_freqs: int = 0
    # For Mixed RoPE, set direction_spacing = None, learnable_rope = True
    direction_spacing: float | None = math.pi * (1 - math.sqrt(5))
    learnable_rope: bool = False
    sep_rope_heads: bool = True
    pooling: Literal["mean", "rotary", "attention"] = "mean"

    n_steps: int = (1_281_167 // 1024) * 90  # == 187_650
    micro_batch_size: int = 256
    valid_every: int = 1_281_167 // 1024
    initial_valid: bool = True
    muon_lr: float = 0.03
    muon_mu: float = 0.95
    muon_wd: float = 0.01
    scalar_lr: float = 0.001
    embeds_lr: float = 0.001
    output_lr: float = 0.001
    adamw_mus: tuple[float, float] = (0.9, 0.95)
    adamw_wd: float = 0.01
    lr_cooldown_start: int | None = (1_281_167 // 1024) * 70
    lr_cooldown_ratio: float = 0.0

    rand_augment: tuple[int, int] = (2, 10)
    mixup: float = 0.2

    do_compile: bool = True
    debug: bool = False
    seed: int = 0
    resume_from: str | None = None


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

        self.scale_temperature = False
        self.register_buffer("temperature", torch.tensor(1.0), persistent=False)

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
        if self.scale_temperature:
            q_NHWhd = q_NHWhd / self.temperature.sqrt()
            k_NHWhd = k_NHWhd / self.temperature.sqrt()
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
        self.image_size = image_size
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
                    min_freq=0.5,
                    max_freq=50.0,
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

        self.output = Output(dim, n_classes, pad_to=64)

    def forward(self, images_NCHW: torch.Tensor) -> torch.Tensor:
        assert torch.is_floating_point(images_NCHW)
        input_NHWD = rearrange(
            images_NCHW,
            "N C (nh ph) (nw pw) -> N nh nw (ph pw C)",
            nh=self.image_size[0] // self.patch_size,
            nw=self.image_size[1] // self.patch_size,
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


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main_process = self.rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        self.log_once(f"running with world_size={self.world_size}")
        if self.world_size > 1:
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            dist.init_process_group(backend="nccl", device_id=self.device)
            if torch.cuda.device_count() != local_world_size:
                log.warning(
                    f"device_count={torch.cuda.device_count()} != {local_world_size=}"
                )

        metrics.enabled = self.is_main_process
        if self.is_main_process and not cfg.debug:
            metrics.use_wandb = True
            wandb.init(
                entity="archibox",
                project="ropend-imagenet-class",
                dir=Path(__file__).parent / "runs/imagenet_class",
                config=cfg.model_dump(),
            )
            log.info(("#" * 40) + "\n" + CODE + "\n" + ("#" * 40))
            log.info(f"using {DTYPE=}")

        nh, nw = 224 // cfg.patch_size, 224 // cfg.patch_size
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
        model = ViTClassifier(
            image_size=(224, 224),
            patch_size=cfg.patch_size,
            dim=cfg.dim,
            mlp_dim=cfg.mlp_dim,
            head_dim=cfg.head_dim,
            depth=cfg.depth,
            n_classes=1000,
            pos_emb=pos_emb,
            pooling=cfg.pooling,
        ).to(self.device)
        self.raw_model = model

        if cfg.do_compile:
            model = torch.compile(model, mode=COMPILE_MODE)
        if self.world_size > 1:
            model = DDP(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        self.model = model

        muon_params, scalar_params, embeds_params, output_params = (
            auto_split_muon_params(
                self.raw_model,
                log_level=logging.INFO if self.is_main_process else logging.DEBUG,
            )
        )
        adamw_params = [
            dict(params=scalar_params, lr=cfg.scalar_lr),
            dict(params=embeds_params, lr=cfg.embeds_lr),
            dict(params=output_params, lr=cfg.output_lr),
        ]
        self.muon = Muon(
            muon_params, lr=cfg.muon_lr, momentum=cfg.muon_mu, weight_decay=cfg.muon_wd
        )
        self.adamw = torch.optim.AdamW(
            adamw_params, betas=cfg.adamw_mus, weight_decay=cfg.adamw_wd
        )
        for opt in (self.muon, self.adamw):
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        if cfg.resume_from is not None:
            ckpt = torch.load(cfg.resume_from)
            train_pipe_ckpt = ckpt["train_pipeline"]
        else:
            train_pipe_ckpt = None

        self.train_pipe = imagenet_train_pipeline(
            checkpoint=train_pipe_ckpt,
            batch_size=cfg.micro_batch_size,
            num_threads=4,
            device_id=self.local_rank,
            num_shards=self.world_size,
            shard_id=self.rank,
            randaug_n=cfg.rand_augment[0],
            randaug_m=cfg.rand_augment[1],
        )
        valid_pipe = imagenet_valid_pipeline(
            batch_size=cfg.micro_batch_size,
            num_threads=4,
            device_id=self.local_rank,
            num_shards=self.world_size,
            shard_id=self.rank,
        )
        self.train_loader = DALIClassificationIterator(
            self.train_pipe,
            reader_name="imagenet_train_reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )
        self.train_loader_iter = iter(self.train_loader)
        self.valid_loader = DALIClassificationIterator(
            valid_pipe,
            reader_name="imagenet_valid_reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )
        log.info(
            f"rank={self.rank}, len(train_loader)={len(self.train_loader_iter):,}, train_loader._size={self.train_loader._size:,}"
        )
        if cfg.mixup > 0:
            self.mixup = v2.MixUp(alpha=cfg.mixup, num_classes=1000)

        if cfg.resume_from is not None:
            self.step = ckpt["step"]
            self.epoch = ckpt["epoch"]
            self.best_valid_loss = ckpt["best_valid_loss"]
            self.raw_model.load_state_dict(ckpt["model"])
            self.muon.load_state_dict(ckpt["muon"])
            self.adamw.load_state_dict(ckpt["adamw"])

            main_rng_size = torch.random.get_rng_state().size(0)
            torch.random.set_rng_state(ckpt["rng"][self.rank, :main_rng_size].cpu())
            torch.cuda.random.set_rng_state(
                ckpt["rng"][self.rank, main_rng_size:].cpu()
            )
        else:
            self.step = 0
            self.epoch = 0
            self.best_valid_loss = float("inf")
            self.last_valid_loss = None

        self.log_once(f"starting @ step={self.step:,}, epoch={self.epoch:,}")

    def log_once(self, msg, level=logging.INFO):
        if self.is_main_process:
            log.log(level, msg)

    def run(self):
        metrics.context = "train_"
        with tqdm.tqdm(
            total=self.cfg.n_steps, desc="training", mininterval=5.0
        ) as progress_bar:
            if self.step == 0:
                if self.cfg.initial_valid:
                    self.log_once("running initial validation epoch")
                    self.valid_epoch()
            else:
                progress_bar.update(self.step)

            while self.step < self.cfg.n_steps:
                # Schedule LR.
                if (
                    self.cfg.lr_cooldown_start is not None
                    and self.cfg.lr_cooldown_start <= self.step
                    and self.cfg.lr_cooldown_start < self.cfg.n_steps
                ):
                    frac = (self.step - self.cfg.lr_cooldown_start + 1) / (
                        self.cfg.n_steps - self.cfg.lr_cooldown_start
                    )
                    relative_lr = 1.0 - (1.0 - self.cfg.lr_cooldown_ratio) * min(
                        1.0, frac
                    )
                    for opt in (self.muon, self.adamw):
                        for group in opt.param_groups:
                            group["lr"] = group["initial_lr"] * relative_lr
                else:
                    relative_lr = 1.0
                metrics.push(relative_lr=relative_lr)

                self.train_step()
                self.step += 1
                progress_bar.update(1)

    def train_step(self):
        metrics.tick("load_batch")
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            self.epoch += 1
            if self.epoch < 5:
                self.log_once(f"starting epoch {self.epoch:,} @ step={self.step}")

            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        images_NCHW = batch[0]["data"]
        labels_N = batch[0]["label"].squeeze(-1).long()
        assert images_NCHW.size(0) == labels_N.size(0) == self.cfg.micro_batch_size
        if self.cfg.mixup > 0:
            images_NCHW, labels_ND = self.mixup(images_NCHW, labels_N)

        metrics.tick("forward")
        logits_ND = self.model(images_NCHW.type(DTYPE))
        assert logits_ND.dtype == DTYPE
        if self.cfg.mixup > 0:
            loss = F.cross_entropy(logits_ND.float(), labels_ND)
            metrics.push(loss=loss)
        else:
            loss = F.cross_entropy(logits_ND.float(), labels_N)
            acc = (torch.argmax(logits_ND, dim=-1) == labels_N).float().mean()
            metrics.push(loss=loss, acc=acc)

        metrics.tick("backward")
        loss.backward()

        metrics.tick("optim")
        for optim in (self.muon, self.adamw):
            optim.step()
        for optim in (self.muon, self.adamw):
            optim.zero_grad(set_to_none=True)

        metrics.tick(None)

        if (self.step + 1) % self.cfg.valid_every == 0:
            self.valid_epoch()

    def valid_epoch(self):
        self.model.eval()
        metrics.context = "valid_"
        with torch.no_grad():
            mean_nll = 0
            mean_acc = 0
            n = 0
            for batch in tqdm.tqdm(
                self.valid_loader, desc="validation", mininterval=5.0
            ):
                images_NCHW = batch[0]["data"]
                labels_N = batch[0]["label"].squeeze(-1).long()

                logits_ND = self.model(images_NCHW.type(DTYPE))
                nll = F.cross_entropy(logits_ND.float(), labels_N)
                acc = (torch.argmax(logits_ND, dim=-1) == labels_N).float().mean()

                n += 1
                mean_nll = mean_nll + (nll - mean_nll) / n
                mean_acc = mean_acc + (acc - mean_acc) / n
            # Reduce validation metrics (not worth speed penalty for train)
            if self.world_size > 1:
                dist.all_reduce(mean_nll, op=dist.ReduceOp.AVG)
                dist.all_reduce(mean_acc, op=dist.ReduceOp.AVG)
            metrics.push(nll=mean_nll, acc=mean_acc)

        if self.is_main_process:
            if "valid_nll" in metrics.mean:
                self.last_valid_loss = metrics.mean["valid_nll"].item()
            else:
                log.warning("valid_loss not found in metrics")

        metrics.report()
        self.save_checkpoint()

        self.model.train()
        metrics.context = "train_"

    def save_checkpoint(self):
        if self.cfg.debug:
            return

        main_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.random.get_rng_state(self.device)
        rng_state = torch.cat([main_rng_state, cuda_rng_state]).to(self.device)
        if self.world_size > 1:
            if self.is_main_process:
                gather_list = [
                    torch.zeros_like(rng_state) for _ in range(self.world_size)
                ]
            else:
                gather_list = None
            dist.gather(rng_state, gather_list, dst=0)
        else:
            gather_list = [rng_state]
        if not self.is_main_process:
            return

        if (
            self.last_valid_loss is not None
            and self.last_valid_loss < self.best_valid_loss
        ):
            self.best_valid_loss = self.last_valid_loss
            save_as_best = True
        else:
            save_as_best = False

        ckpt = dict(
            step=self.step,
            epoch=self.epoch,
            best_valid_loss=self.best_valid_loss,
            train_pipeline=self.train_pipe.checkpoint(),
            model=self.raw_model.state_dict(),
            muon=self.muon.state_dict(),
            adamw=self.adamw.state_dict(),
            rng=torch.stack(gather_list),
        )
        if save_as_best:
            self.log_once(f"saving best @ step={self.step:,}, epoch={self.epoch:,}")
            ckpt_names = ["latest", "best"]
        else:
            ckpt_names = ["latest"]
        for name in ckpt_names:
            savepath = Path(wandb.run.dir) / f"{name}.ckpt.new"
            torch.save(ckpt, savepath)
            savepath.replace(Path(wandb.run.dir) / f"{name}.ckpt")


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
