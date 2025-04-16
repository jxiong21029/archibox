import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange
from PIL import Image
from torch.distributions import Categorical

from archibox.components import FusedLinear, RMSNorm, make_embedding
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


@dataclass
class MnistVectorFSQConfig:
    img_mean: float = 0.1307
    img_std: float = 0.3081

    dim: int = 1024
    mlp_dim: int = 2048

    inpaint_size: int = 8

    latent_ndim: int = 10
    latent_bins: int = 4
    latent_group_size: int = 5

    cond_encoder_depth: int = 4
    fsq_prior_blocks: int = 2
    fsq_decoder_depth: int = 4

    mse_weight_sigma: float = 0.05


@dataclass
class Config:
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/fsq_inpaint")

    model: MnistVectorFSQConfig = field(default_factory=MnistVectorFSQConfig)
    do_compile: bool = False
    n_steps: int = 2000
    valid_every: int | None = 100
    batch_size: int = 4096

    muon_lr: float = 0.01
    muon_mu: float = 0.9
    muon_wd: float = 0.03
    scalar_lr: float = 0.0005
    embeds_lr: float = 0.0005
    output_lr: float = 0.0005
    adamw_mu1: float = 0.9
    adamw_mu2: float = 0.99
    adamw_wd: float = 0.03
    lr_cooldown_start: int | None = None
    lr_cooldown_ratio: float = 0.0


class FSQ(nn.Module):
    def __init__(self, dim_in: int, ndim: int, bins: int):
        super().__init__()
        self.bins = bins
        self.in_proj = nn.Sequential(
            RMSNorm(dim_in, affine=False),
            FusedLinear(dim_in, ndim),
        )

    def forward(self, x):
        z = torch.tanh(self.in_proj(x).float())  # in [-1, 1]
        z = (z + 1.0) * ((self.bins - 1) / 2.0)  # [-1, 1] --> [0, nbins - 1]
        idxs = torch.round(z.detach())
        z = z + (idxs - z).detach()  # round w/ straight-through grad
        z = z * (2.0 / (self.bins - 1)) - 1.0  # [0, nbins - 1] --> [-1, 1]
        return z.type_as(x), idxs.type(torch.long)


class VectorModulatedMLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.wx = FusedLinear(dim, mlp_dim)
        self.wy = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.wv = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, y):
        scores = self.act(self.wx(self.norm(x)) * self.wy(self.norm(y)))
        return self.wv(scores)


class EmbedModulatedMLP(nn.Module):
    def __init__(self, dim, mlp_dim, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.wq = FusedLinear(dim, mlp_dim)
        self.k = make_embedding(vocab_size, mlp_dim)
        self.act = nn.SiLU()
        self.wv = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, ids):
        scores = self.act(self.wq(self.norm(x)) * self.k(ids))
        return self.wv(scores)


class VectorFSQ(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        data_dim: int,
        latent_ndim: int,
        latent_bins: int,
        latent_group_size: int,
        prior_blocks: int,
        decoder_blocks: int,
    ):
        super().__init__()
        assert latent_ndim % latent_group_size == 0
        self.latent_ndim = latent_ndim
        self.latent_bins = latent_bins
        self.latent_group_size = latent_group_size
        self.ngroups = latent_ndim // latent_group_size
        assert self.ngroups >= 1

        self.prior_mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(dim, affine=False),
                    FusedLinear(dim, mlp_dim),
                    nn.SiLU(),
                    FusedLinear(mlp_dim, dim, scale=True, zero_init=True),
                )
                for _ in range(prior_blocks)
            ]
        )

        self.vocab_size = latent_bins**latent_group_size
        self.prior_heads = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(dim, affine=False),
                    FusedLinear(dim, self.vocab_size, zero_init=True),
                )
                for _ in range(self.ngroups)
            ]
        )
        for head in self.prior_heads:
            head[-1]._is_output = True

        self.prior_blocks = nn.ModuleList(
            [
                EmbedModulatedMLP(dim, mlp_dim, self.vocab_size)
                for _ in range(self.ngroups - 1)
            ]
        )

        self.encoder_in = FusedLinear(data_dim, dim)
        self.encoder = VectorModulatedMLP(dim, mlp_dim)

        self.fsq = FSQ(dim, latent_ndim, latent_bins)
        self.fsq_out = FusedLinear(latent_ndim, dim)

        self.decoder_blocks = nn.ModuleList(
            [VectorModulatedMLP(dim, mlp_dim) for _ in range(decoder_blocks)]
        )
        self.decoder_head = nn.Sequential(
            RMSNorm(dim, affine=False), FusedLinear(dim, data_dim, zero_init=True)
        )
        self.decoder_head._is_output = True

    def forward(self, x, y):
        latents, raw_ids = self.fsq(self.encoder(x, self.encoder_in(y)))
        assert raw_ids.size(-1) == self.ngroups * self.latent_group_size
        group_ids = (
            raw_ids.reshape(raw_ids.shape[:-1] + (self.ngroups, self.latent_group_size))
            * (
                self.latent_bins
                ** torch.arange(self.latent_group_size, device=x.device)
            )
        ).sum(dim=-1)
        assert group_ids.size(-1) == self.ngroups

        h = x
        for block in self.prior_mlp_blocks:
            h = h + block(h)
        logits = [self.prior_heads[0](h)]
        for i, block in enumerate(self.prior_blocks):
            h = h + block(h, group_ids[..., i])
            logits.append(self.prior_heads[i + 1](h))
        logits = torch.stack(logits, dim=-2)
        xent = F.cross_entropy(logits.flatten(0, -2), group_ids.flatten())

        yhat = self.fsq_out(latents * (3**0.5))
        for block in self.decoder_blocks:
            yhat = yhat + block(x, yhat)
        yhat = self.decoder_head(yhat)

        mse = (yhat - y).pow(2).mean()
        return xent, mse

    def generate(self, x):
        group_ids = []

        h = x
        for block in self.prior_mlp_blocks:
            h = h + block(h)
        logits = self.prior_heads[0](h)
        group_ids.append(Categorical(logits=logits).sample())
        for i, block in enumerate(self.prior_blocks):
            h = h + block(h, group_ids[-1])
            logits = self.prior_heads[i + 1](h)
            group_ids.append(Categorical(logits=logits).sample())
        group_ids = torch.stack(group_ids, dim=-1)

        assert group_ids.size(-1) == self.ngroups
        raw_ids = torch.zeros(
            group_ids.shape[:-1] + (self.ngroups, self.latent_group_size),
            device=group_ids.device,
            dtype=torch.long,
        )
        for i in range(self.latent_group_size):
            raw_ids[..., i] = group_ids % self.latent_bins
            group_ids = group_ids // self.latent_bins
        raw_ids = raw_ids.flatten(-2, -1)
        assert raw_ids.size(-1) == self.latent_ndim
        latents = raw_ids.type(DTYPE) * (2 / (self.latent_bins - 1)) - 1
        yhat = self.fsq_out(latents * (3**0.5))
        for block in self.decoder_blocks:
            yhat = yhat + block(x, yhat)
        yhat = self.decoder_head(yhat)
        return yhat


class MnistVectorFSQ(nn.Module):
    def __init__(self, cfg: MnistVectorFSQConfig, metrics):
        super().__init__()
        self.cfg = cfg
        self.metrics = metrics

        xy = torch.arange(28 * 28)
        self.mask = nn.Buffer(
            ((xy // 28) < 14 - (cfg.inpaint_size // 2))
            | ((xy // 28) >= 14 + (cfg.inpaint_size // 2))
            | ((xy % 28) < 14 - (cfg.inpaint_size // 2))
            | ((xy % 28) >= 14 + (cfg.inpaint_size // 2)),
            persistent=False,
        )

        self.cond_encoder_in = FusedLinear(28 * 28, cfg.dim)
        self.cond_encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(cfg.dim, affine=False),
                    FusedLinear(cfg.dim, cfg.mlp_dim),
                    nn.SiLU(),
                    FusedLinear(cfg.mlp_dim, cfg.dim),
                )
                for _ in range(cfg.cond_encoder_depth)
            ]
        )
        self.vector_fsq = VectorFSQ(
            cfg.dim,
            cfg.mlp_dim,
            data_dim=self.cfg.inpaint_size**2,
            latent_ndim=cfg.latent_ndim,
            latent_bins=cfg.latent_bins,
            latent_group_size=cfg.latent_group_size,
            prior_blocks=cfg.fsq_prior_blocks,
            decoder_blocks=cfg.fsq_decoder_depth,
        )

    def forward(self, images, labels):
        N = images.size(0)
        images = images.type(DTYPE) / 255.0
        images = images.flatten(1, -1)
        images = (images - self.cfg.img_mean) / self.cfg.img_std
        cond = images * self.mask
        assert images.size(-1) == 28 * 28
        lo = 14 - self.cfg.inpaint_size // 2
        hi = 14 + self.cfg.inpaint_size // 2
        y = images.reshape(N, 28, 28)[:, lo:hi, lo:hi].reshape(
            N, self.cfg.inpaint_size**2
        )

        x = self.cond_encoder_in(cond)
        for block in self.cond_encoder_blocks:
            x = x + block(x)

        xent, mse = self.vector_fsq(x, y)
        loss = xent + mse / self.cfg.mse_weight_sigma
        self.metrics.push(xent=xent, mse=mse, loss=loss)
        return loss

    def generate(self, images):
        images = images.type(DTYPE) / 255.0
        images = images.flatten(1, -1)
        images = (images - self.cfg.img_mean) / self.cfg.img_std
        cond = images * self.mask
        x = self.cond_encoder_in(cond)
        y = self.vector_fsq.generate(x)
        return y


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
        torch.manual_seed(self.rank)
        torch.cuda.manual_seed(self.rank)

        self.metrics = Metrics(enabled=self.is_main_process, use_wandb=cfg.use_wandb)
        self.metrics.context = "train_"

        dataset_path = Path(__file__).parent / "data"
        dataset_path.mkdir(exist_ok=True)

        self.train_loader = mnist_loader(
            train=True, batch_size=cfg.batch_size, device=self.device
        )
        self.model = MnistVectorFSQ(cfg.model, self.metrics).to(self.device)
        if cfg.do_compile:
            self.model = torch.compile(self.model)

        scalar_params = []
        embeds_params = []
        output_params = []
        muon_params = []
        for name, p in self.model.named_parameters():
            shape = tuple(p.shape)
            if not p.requires_grad:
                self.debug_once(f"{name} {shape} requires_grad=False, skipped")
                continue
            elif p.ndim < 2:
                self.debug_once(f"{name} {shape} assigned to AdamW")
                scalar_params.append(p)
            elif hasattr(p, "_is_embed") and p._is_embed:
                self.debug_once(f"{name} {shape} (_is_embed=True) assigned to AdamW")
                embeds_params.append(p)
            elif hasattr(p, "_is_output") and p._is_output:
                self.debug_once(f"{name} {shape} (_is_output=True) assigned to AdamW")
                output_params.append(p)
            else:
                if hasattr(p, "_ortho") and self.is_main_process:
                    log.warning(
                        "_ortho is deprecated, use _is_embed or _is_output instead"
                    )
                self.debug_once(f"{name}{shape} assigned to Muon")
                muon_params.append(p)
        if self.is_main_process:
            total_params = sum(
                p.numel()
                for p in muon_params + scalar_params + embeds_params + output_params
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
            dict(params=scalar_params, lr=self.cfg.scalar_lr),
            dict(params=embeds_params, lr=self.cfg.embeds_lr),
            dict(params=output_params, lr=self.cfg.output_lr),
        ]

        self.muon = Muon(
            muon_params,
            lr=self.cfg.muon_lr,
            momentum=self.cfg.muon_mu,
            weight_decay=self.cfg.muon_wd,
        )
        self.adamw = torch.optim.AdamW(
            adamw_params,
            betas=[self.cfg.adamw_mu1, self.cfg.adamw_mu2],
            weight_decay=self.cfg.adamw_wd,
        )

        self.optims = [self.muon, self.adamw]
        for opt in self.optims:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        self.step = 0

    def log_once(self, s):
        if self.is_main_process:
            log.info(s)

    def debug_once(self, s):
        if self.is_main_process:
            log.debug(s)

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
        self.metrics.push(relative_lr=relative_lr)

    def train_step(self):
        self.schedule_lr()

        images, labels = next(self.train_loader)
        h, v = torch.rand(2)
        if h < 0.25:
            images = F.pad(images, (1, -1))
        elif h > 0.75:
            images = F.pad(images, (-1, 1))
        if v < 0.25:
            images = F.pad(images, (0, 0, 1, -1))
        elif v > 0.75:
            images = F.pad(images, (0, 0, -1, 1))

        loss = self.model(images, labels)
        loss.backward()
        for optim in self.optims:
            optim.step()
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

    @torch.no_grad
    def valid_epoch(self):
        self.model.eval()
        self.metrics.context = "valid_"
        for images, labels in mnist_loader(
            train=False, batch_size=self.cfg.batch_size, epochs=1, device=self.device
        ):
            self.model(images, labels)

        # Generate and save samples
        truth = {c: [] for c in range(10)}
        rows = 5
        count = 0
        for img, label in mnist_loader(
            train=False, batch_size=1, epochs=1, device="cpu"
        ):
            if len(truth[int(label)]) < rows:
                truth[int(label)].append(img.squeeze(0).numpy())
                count += 1
                if count == 10 * rows:
                    break
        truth = [truth[c][i] for c in range(10) for i in range(rows)]
        truth = np.array(truth)

        if self.step == 0:
            Image.fromarray(rearrange(truth, "(D M) H W -> (M H) (D W)", D=10)).save(
                Path(__file__).parent / "data/truth.png"
            )

        model_inputs = torch.tensor(truth, device=self.device)
        inpaint = self.model.generate(model_inputs)
        inpaint = rearrange(
            inpaint,
            "N (h w) -> N h w",
            h=self.model.cfg.inpaint_size,
            w=self.model.cfg.inpaint_size,
        )
        inpaint = inpaint * self.model.cfg.img_std + self.model.cfg.img_mean
        inpaint = inpaint.clamp(0, 1) * 255
        inpaint = inpaint.type(torch.uint8).cpu().numpy()
        inpainted = np.copy(truth)
        lo = 14 - self.model.cfg.inpaint_size // 2
        hi = 14 + self.model.cfg.inpaint_size // 2
        inpainted[:, lo:hi, lo:hi] = inpaint

        savedir = Path(self.cfg.savedir)
        savedir.mkdir(exist_ok=True)
        if self.step > 0:
            savepath = savedir / f"samples_step_{self.step + 1}.png"
        else:
            savepath = savedir / "samples_initial.png"
        Image.fromarray(rearrange(inpainted, "(D M) H W -> (M H) (D W)", D=10)).save(
            savepath
        )

        self.metrics.report()
        self.model.train()
        self.metrics.context = "train_"

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


def main():
    logging.basicConfig(level=logging.INFO)

    cfg = Config()
    try:
        trainer = Trainer(cfg)
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
