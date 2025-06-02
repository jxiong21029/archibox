import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from einops import rearrange
from PIL import Image
from torch import Tensor

from archibox.components import FusedLinear, RMSNorm
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.mnist_gen.fid import FID
from archibox.muon import Muon
from archibox.utils import parse_config

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


@dataclass
class FlowConfig:
    dim: int = 1024
    mlp_dim: int = 2048
    freq_dim: int = 1024
    min_freq: float = 2 * torch.pi
    max_freq: float = 200 * torch.pi
    depth: int = 4

    img_mean: float = 0.5
    img_std: float = 0.5

    solver: str = "analytic"
    noise_ratio: float = 1.0
    sampling_steps: int = 512


@dataclass
class Config:
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/vector_flow")

    model: FlowConfig = field(default_factory=FlowConfig)
    do_compile: bool = False
    n_steps: int = 1_000
    valid_every: int | None = 1000
    batch_size: int = 8192

    use_muon: bool = True
    muon_lr: float = 0.02
    adamw_lr: float = 0.003
    wd: float = 0.01
    lr_cooldown_start: int | None = None
    lr_cooldown_ratio: float = 0.0


def make_embedding(num_embeddings: int, embedding_dim: int):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    embed.to(dtype=DTYPE)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


class ConditionalMLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.k = FusedLinear(dim, mlp_dim)
        self.wc = FusedLinear(dim, dim, zero_init=True)
        self.act = nn.SiLU()
        self.v = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, c):
        scores = self.k(self.norm(x) * (1 + self.wc(self.norm(c))))
        scores = self.act(scores)
        return self.v(scores)


class FlatTrigflow(nn.Module):
    """A modified version of trigflow for t in [0, 1]. Designed for flat vector data."""

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        data_dim: int,
        depth: int,
        min_freq: float,
        max_freq: float,
        freq_dim: int,
    ):
        """
        Args:
            dim: dimensionality of main path
            mlp_dim: dimensionality of the MLP blocks' hidden activations
            data_dim: dimensionality of the input
            depth: number of MLP blocks. total # of linear layers = 2 * depth + 2
            min_freq: minimum frequency of the sinusoidal embedding for time
                suggested: value between 1 and 2pi
            max_freq: maximum frequency of the sinusoidal embedding for time
                suggested: 100 * min_freq
            freq_dim: dimensionality of the sinusoidal embedding for time
                suggested: a power of two >= 256
        """

        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.data_dim = data_dim
        self.freq_dim = freq_dim
        self.min_freq = min_freq
        self.max_mult = max_freq / min_freq

        self.in_proj = FusedLinear(data_dim, dim, bias=True)
        self.t_proj = FusedLinear(freq_dim * 2, dim)
        self.blocks = nn.ModuleList(
            [ConditionalMLPBlock(dim, mlp_dim) for _ in range(depth)]
        )
        self.out_head = nn.Sequential(
            RMSNorm(dim, affine=False), FusedLinear(dim, data_dim, zero_init=True)
        )
        self.out_head[-1].weight._is_output = True

    def time_embed(self, t: Tensor) -> Tensor:
        """Sinusoidal embedding for time. Expects input of shape (..., 1)"""
        assert t.size(-1) == 1
        freqs = self.min_freq * (
            self.max_mult ** torch.linspace(0, 1, self.freq_dim, device=t.device)
        )
        thetas = t.float() * freqs.float()
        return torch.cat([thetas.sin(), thetas.cos()], dim=-1)

    def loss(self, inputs: Tensor, cond: Tensor) -> Tensor:
        assert inputs.size(-1) == self.data_dim
        assert cond.size(-1) == self.dim
        device, dtype = inputs.device, inputs.dtype

        x1 = inputs.float()
        t = torch.sigmoid(torch.randn(inputs.shape[:-1] + (1,), device=device))
        t_emb = self.t_proj(self.time_embed(t).type(dtype))

        x0 = torch.randn_like(x1)
        s = t * (torch.pi / 2)
        xt = torch.cos(s) * x0 + torch.sin(s) * x1
        vt = (-torch.sin(s) * x0 + torch.cos(s) * x1) * (torch.pi / 2)

        c = t_emb + cond
        pt = self(xt.type(dtype), t, c)

        loss = (pt - vt).pow(2)
        return loss

    def forward(self, xt: Tensor, t: Tensor, c: Tensor) -> Tensor:
        h = self.in_proj(xt * t.pow(0.5).type_as(xt))
        # h = self.in_proj(xt)
        t_emb = self.t_proj(self.time_embed(t).type_as(h))
        c = c + t_emb
        for block in self.blocks:
            h = h + block(h, c)
        return self.out_head(h).float()

    @torch.no_grad
    def sample_heun(self, cond: Tensor, noise_ratio: float, n_steps: int) -> Tensor:
        """Samples using Heun's method as a solver for the PF-ODE.

        Args:
            cond: conditioning input
            noise_ratio: ratio of the amount of noise added to the progress per step.
                In each step, we take a reverse step (removing noise), then a forward
                step (adding noise back), akin to Langevin dynamics. Set to 0 for
                deterministic sampling.
            n_steps: total # of heun steps == (total # of function evaluations / 2)
        """

        device, dtype = cond.device, cond.dtype

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt = torch.randn(cond.shape[:-1] + (self.data_dim,), device=device)
        noise_dt = noise_ratio / n_steps

        for i in range(n_steps):
            ta = timesteps[i].view(1)
            tb = timesteps[i + 1].view(1)
            tc = (timesteps[i + 1].view(1) + noise_dt).clamp(max=1)
            sb = tb * (torch.pi / 2)
            sc = tc * (torch.pi / 2)

            va = self(xt.type(dtype), ta, cond)
            xc = xt + (tc - ta) * va
            vc = self(xc.type(dtype), tc, cond)
            xc = xt + (tc - ta) * (va + vc) / 2

            if sb < sc:
                scale = sb.sin() / sc.sin()
                extra_var = sb.cos().pow(2) - (sc.cos() * scale).pow(2)
                xt = scale * xc + extra_var.sqrt() * torch.randn_like(xt)
            else:
                xt = xc
        return xt

    @torch.no_grad
    def sample_analytic(self, cond: Tensor, noise_ratio: float, n_steps: int) -> Tensor:
        """Solves for the next timestep's xt using the forward process equation.

        Leverages the fact that we know xt = sin(s) x + cos(s) z and vt = pi/2 cos(s) x
        - pi/2 sin(s) z. We can plug in the value of xt and the model's estimate of vt
        to solve for x and z, and then compute the estimated value of xt' for any t'.

        Args:
            cond: conditioning input
            noise_ratio: ratio of the amount of noise added to the progress per step.
                In each step, we take a reverse step (from ta -> tc, removing noise),
                then a forward step (from tc -> tb, adding noise back), akin to Langevin
                dynamics. Set noise_ratio to 0 for deterministic sampling.
            n_steps: total # of model evaluations
        """
        device, dtype = cond.device, cond.dtype

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt = torch.randn(cond.shape[:-1] + (self.data_dim,), device=device)
        noise_dt = noise_ratio / n_steps

        for i in range(n_steps):
            ta = timesteps[i].view(1)
            tb = timesteps[i + 1].view(1)
            tc = (timesteps[i + 1].view(1) + noise_dt).clamp(max=1)
            sa = ta * (torch.pi / 2)
            sb = tb * (torch.pi / 2)
            sc = tc * (torch.pi / 2)

            va = self(xt.type(dtype), ta, cond)
            xc = torch.cos(sa - sc) * xt - (2 / torch.pi) * torch.sin(sa - sc) * va
            if sb < sc:
                scale = sb.sin() / sc.sin()
                extra_var = sb.cos().pow(2) - (sc.cos() * scale).pow(2)
                xt = scale * xc + extra_var.sqrt() * torch.randn_like(xt)
            else:
                xt = xc
        return xt


class MnistFlow(nn.Module):
    def __init__(self, cfg: FlowConfig, metrics: Metrics):
        assert cfg.solver in ("analytic", "heun")
        super().__init__()
        self.cfg = cfg
        self.metrics = metrics
        self.flow = FlatTrigflow(
            dim=cfg.dim,
            mlp_dim=cfg.mlp_dim,
            data_dim=28 * 28,
            freq_dim=cfg.freq_dim,
            depth=cfg.depth,
            min_freq=cfg.min_freq,
            max_freq=cfg.max_freq,
        )
        self.class_embed = make_embedding(10, cfg.dim)

    def forward(self, images_NHW, labels_N):
        images_N1HW = images_NHW.type(DTYPE) / 255.0
        imgs_ND = images_N1HW.flatten(1, -1)
        imgs_ND = (imgs_ND - self.cfg.img_mean) / self.cfg.img_std

        class_emb_ND = self.class_embed(labels_N).type(DTYPE)

        loss = self.flow.loss(imgs_ND, class_emb_ND).mean()
        self.metrics.push(loss=loss)
        return loss

    @torch.no_grad
    def generate(self, labels_N):
        class_emb_ND = self.class_embed(labels_N).type(DTYPE)
        if self.cfg.solver == "analytic":
            sample_fn = self.flow.sample_analytic
        else:
            assert self.cfg.solver == "heun"
            sample_fn = self.flow.sample_heun
        imgs_ND = sample_fn(
            class_emb_ND,
            noise_ratio=self.cfg.noise_ratio,
            n_steps=self.cfg.sampling_steps,
        )
        return imgs_ND


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

        self.metrics = Metrics(enabled=self.is_main_process, use_wandb=cfg.use_wandb)
        self.metrics.context = "train_"

        dataset_path = Path(__file__).parent / "data"
        dataset_path.mkdir(exist_ok=True)

        self.train_loader = mnist_loader(
            train=True, batch_size=cfg.batch_size, device=self.device
        )
        self.model = MnistFlow(cfg.model, self.metrics).to(self.device)
        if cfg.do_compile:
            self.model = torch.compile(self.model)

        muon_params = []
        adamw_params = []
        for name, p in self.model.named_parameters():
            shape = tuple(p.shape)
            if not p.requires_grad:
                self.log_once(
                    f"{name} {shape} requires_grad=False, skipped", logging.DEBUG
                )
                continue
            elif (
                p.ndim < 2
                or not cfg.use_muon
                or (hasattr(p, "_is_embed") and p._is_embed)
                or (hasattr(p, "_is_output") and p._is_output)
            ):
                self.log_once(f"{name} {shape} assigned to AdamW", logging.DEBUG)
                adamw_params.append(p)
            else:
                self.log_once(f"{name} {shape} assigned to Muon", logging.DEBUG)
                muon_params.append(p)

        if self.is_main_process:
            total_params = sum(p.numel() for p in muon_params + adamw_params)
            total_param_tensors = sum(
                len(group) for group in (muon_params, adamw_params)
            )
            log.info(
                "parameter information:\n"
                f"- muon params: {sum(p.numel() for p in muon_params):,} over {len(muon_params):,} tensors\n"
                f"- adamw params: {sum(p.numel() for p in adamw_params):,} over {len(adamw_params):,} tensors\n"
                f"total: {total_params:,} over {total_param_tensors:,} tensors"
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
            adamw_params,
            lr=self.cfg.adamw_lr,
            betas=[0.9, 0.99],
            weight_decay=self.cfg.wd,
        )
        self.optims.append(adamw)
        for opt in self.optims:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        self.step = 0

        self.fid = FID.from_pretrained()
        self.fid.model.to(self.device)

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
        self.metrics.push(relative_lr=relative_lr)

    def train_step(self):
        self.schedule_lr()

        images, labels = next(self.train_loader)
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

        rows = 5
        # Generate and save samples
        labels_N = torch.arange(10 * rows, device=self.device) // rows
        imgs_ND = self.model.generate(labels_N)
        imgs = rearrange(imgs_ND, "N (H W) -> N H W", H=28, W=28)
        imgs = imgs * self.model.cfg.img_std + self.model.cfg.img_mean
        imgs = imgs.clamp(0, 1) * 255
        imgs = imgs.type(torch.uint8)
        imgs = imgs.cpu().numpy()

        savedir = Path(self.cfg.savedir)
        savedir.mkdir(exist_ok=True)
        if self.step > 0:
            savepath = savedir / f"samples_step_{self.step + 1}.png"
        else:
            savepath = savedir / "samples_initial.png"

        imgs = rearrange(imgs, "(D M) H W -> (M H) (D W)", D=10)
        Image.fromarray(imgs).save(savepath)

        if self.step == 0:
            truth = {c: [] for c in range(10)}
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
            truth = rearrange(truth, "(D M) H W -> (M H) (D W)", D=10)
            Image.fromarray(truth).save(Path(__file__).parent / "data/truth.png")

        # Compute FID-like metric using features from an MLP classifier
        total = 50_000
        samples = []
        for i in tqdm.trange(0, total, self.cfg.batch_size, desc="computing fid"):
            labels_N = torch.arange(i, i + self.cfg.batch_size, device=self.device) % 10
            imgs_ND = self.model.generate(labels_N)
            imgs_ND = imgs_ND * self.model.cfg.img_std + self.model.cfg.img_mean
            imgs_ND = imgs_ND.clamp(0, 1) * 2.0 - 1.0
            samples.append(imgs_ND)
        samples_ND = torch.cat(samples)
        fid = self.fid.compute_fid(samples_ND)
        self.metrics.push(fid=fid)

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
