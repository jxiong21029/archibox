import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from einops import rearrange
from PIL import Image

from archibox.components import FusedLinear, RMSNorm
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


@dataclass
class FlowConfig:
    dim: int = 768
    mlp_dim: int = 2048
    embed_dim: int = 128
    depth: int = 3

    flow_variant: str = "rectified"
    sampling: str = "logit"
    img_mean: float = 0.1307
    img_std: float = 0.3081


@dataclass
class Config:
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/flow_rectified")

    model: FlowConfig = field(default_factory=FlowConfig)
    do_compile: bool = False
    n_steps: int = 20_000
    valid_every: int | None = 1000
    batch_size: int = 2048

    muon_lr: float = 0.03
    muon_mu: float = 0.95
    muon_wd: float = 0.01
    scalar_lr: float = 0.0015
    embeds_lr: float = 0.0015
    output_lr: float = 0.0015
    adamw_mu1: float = 0.9
    adamw_mu2: float = 0.99
    adamw_wd: float = 0.01
    lr_cooldown_start: int | None = 10_000
    lr_cooldown_ratio: float = 0.0


def make_embedding(num_embeddings: int, embedding_dim: int):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    embed.to(dtype=DTYPE)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


class ModulatedMLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.wk = FusedLinear(dim, mlp_dim)
        self.wc = FusedLinear(cond_dim, mlp_dim)
        self.act = nn.SiLU()
        self.wv = FusedLinear(mlp_dim, dim, zero_init=True)

    def forward(self, x, c):
        scores = self.wk(self.norm(x))
        scores = self.act(scores * self.wc(c))
        return x + self.wv(scores)


class Flow(nn.Module):
    def __init__(self, cfg: FlowConfig, metrics: Metrics):
        super().__init__()
        self.cfg = cfg
        self.metrics = metrics
        self.class_embed = make_embedding(10, cfg.embed_dim)
        self.in_proj = FusedLinear(28 * 28, cfg.dim)
        self.blocks = nn.ModuleList(
            [
                ModulatedMLP(cfg.dim, cfg.mlp_dim, cond_dim=cfg.embed_dim)
                for _ in range(cfg.depth)
            ]
        )
        self.out_proj = nn.Sequential(
            RMSNorm(cfg.dim, affine=True), FusedLinear(cfg.dim, 28 * 28, zero_init=True)
        )
        self.out_proj[-1].weight._is_output = True

    def time_embed(self, t_N1, freq_lo=1.0, freq_hi=16.0):
        freqs_D = freq_lo * (
            freq_hi
            ** torch.linspace(0, 1, self.cfg.embed_dim // 2, device=t_N1.device).float()
        )
        theta_ND = t_N1.float() * freqs_D
        return torch.cat([theta_ND.sin(), theta_ND.cos()], dim=1)

    def forward(self, images_NHW, labels_N):
        N, _, _ = images_NHW.shape
        assert images_NHW.dtype == torch.uint8
        assert labels_N.shape == (N,)
        images_N1HW = images_NHW.type(DTYPE) / 255.0
        imgs_ND = images_N1HW.flatten(1, -1)
        imgs_ND = (imgs_ND - self.cfg.img_mean) / self.cfg.img_std

        if self.cfg.sampling == "logit":
            t_N = torch.sigmoid(torch.randn(N, device=imgs_ND.device) * 1.2)
        else:
            assert self.cfg.sampling == "uniform"
            t_N = torch.rand(N, device=imgs_ND.device)
        t_N1 = t_N.unsqueeze(1)
        t_emb_ND = self.time_embed(t_N1).type(DTYPE)
        c_ND = self.class_embed(labels_N).type(DTYPE) + t_emb_ND

        x0_ND = torch.randn_like(imgs_ND)
        if self.cfg.flow_variant == "rectified":
            xt_ND = (1 - t_N1.type(DTYPE)) * x0_ND + t_N1.type(DTYPE) * imgs_ND
            vt_ND = imgs_ND - x0_ND
        else:
            assert self.cfg.flow_variant == "trig"
            theta_N1 = t_N1 * (math.pi / 2)
            xt_ND = (
                torch.cos(theta_N1).type(DTYPE) * x0_ND
                + torch.sin(theta_N1).type(DTYPE) * imgs_ND
            )
            vt_ND = (
                torch.cos(theta_N1).type(DTYPE) * imgs_ND - torch.sin(theta_N1) * x0_ND
            )

        h_ND = self.in_proj(xt_ND)
        for block in self.blocks:
            h_ND = block(h_ND, c_ND)
        p_ND = self.out_proj(h_ND)

        loss = (p_ND - vt_ND).pow(2).mean()
        self.metrics.push(p_std=p_ND.std(), v_std=vt_ND.std(), loss=loss)
        return loss

    @torch.no_grad
    def generate(self, labels_N, n_steps: int = 64):
        (N,) = labels_N.shape
        device = labels_N.device

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt_ND = torch.randn(N, 28 * 28, device=device)
        c_ND = self.class_embed(labels_N).type(DTYPE)

        for t in range(n_steps):
            t_start_N = timesteps[t].view(1).expand(N)
            t_stop_N = timesteps[t + 1].view(1).expand(N)
            t_start_emb_ND = self.time_embed(t_start_N.unsqueeze(1)).type(DTYPE)
            t_stop_emb_ND = self.time_embed(t_stop_N.unsqueeze(1)).type(DTYPE)

            # Heun sampling
            h_ND = self.in_proj(xt_ND.type(DTYPE))
            for block in self.blocks:
                h_ND = block(h_ND, c_ND + t_start_emb_ND)
            p_ND = self.out_proj(h_ND).float()

            xt2_ND = xt_ND + (t_stop_N - t_start_N).unsqueeze(1) * p_ND
            h_ND = self.in_proj(xt2_ND.type(DTYPE))
            for block in self.blocks:
                h_ND = block(h_ND, c_ND + t_stop_emb_ND)
            p2_ND = self.out_proj(h_ND).float()

            xt_ND = xt_ND + (t_stop_N - t_start_N).unsqueeze(1) * (p_ND + p2_ND) / 2
        return xt_ND


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
        self.model = Flow(cfg.model, self.metrics).to(self.device)
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
