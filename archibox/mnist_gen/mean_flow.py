import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms.v2 as T
import tqdm
from bnnp import Metrics, parse_config
from bnnp.nn import FusedLinear, Output, RMSNorm
from einops import rearrange
from PIL import Image
from pydantic import BaseModel, Field
from torch import Tensor
from torch.func import jvp
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

from archibox.mnist_gen.fid import FID
from archibox.trainer import Trainer, TrainerConfig

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32
metrics = Metrics(enabled=False, use_wandb=False, use_cuda_events=True)


class FlowConfig(BaseModel):
    schedule: Literal["trig", "linear"] = "trig"
    lognorm_mu: float = 0.0
    lognorm_std: float = 1.0
    loss_wt_power: float = 0.0
    consistency_steps: int = 0
    loss_consistency_wt: float = 0.0

    sample_shape: list[int] = (28 * 28,)
    sampling_steps: int = 32


class MLPConfig(BaseModel):
    data_dim: int = 28 * 28
    dim: int = 1024
    mlp_dim: int = 2048
    freq_dim: int = 256
    # norm_fourier_freqs: tuple[float, float] = (0.1, 100.0)
    time_fourier_freqs: tuple[float, float] = (0.1, 100.0)
    diff_fourier_freqs: tuple[float, float] = (0.1, 100.0)
    depth: int = 4
    use_adaln: bool = True


SAVEDIR = Path(__file__).parent / "data/meanflow"


class Config(BaseModel):
    use_wandb: bool = False

    net: MLPConfig = Field(default_factory=MLPConfig)
    flow: FlowConfig = Field(default_factory=FlowConfig)
    trainer: TrainerConfig = Field(
        default_factory=lambda: TrainerConfig(
            run_dir=None,
            n_steps=1_000,
            valid_every=250,
            save_every=None,
            micro_batch_size=8192,
            train_loader_workers=0,
            valid_loader_workers=0,
            muon_lr=0.02,
            muon_wd=0.01,
            scalar_lr=0.003,
            embeds_lr=0.003,
            output_lr=0.003,
            adamw_wd=0.01,
            adamw_mu1=0.9,
            adamw_mu2=0.99,
        )
    )


def make_embedding(num_embeddings: int, embedding_dim: int):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    embed.to(dtype=DTYPE)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


class GatedMLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.modulation = FusedLinear(dim, dim, zero_init=True)
        self.k = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.v = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, c):
        gate = self.modulation(self.norm(c))
        scores = self.k(self.norm(x) * (1 + gate))
        scores = self.act(scores)
        return x + self.v(scores)


class AdaLNMLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.dim = dim

        self.norm = RMSNorm(dim)
        self.modulation = FusedLinear(dim, 3 * dim, zero_init=True)
        self.k = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.v = FusedLinear(mlp_dim, dim)

    def forward(self, x, c):
        gate, shift, scale = self.modulation(self.norm(c)).chunk(3, dim=-1)
        scores = self.k(self.norm(x) * (1.0 + gate) + shift)
        scores = self.act(scores)
        return x + self.v(scores) * scale


def make_frequencies(freq_range: tuple[float, float], dim: int):
    min_freq, max_freq = freq_range
    return min_freq * ((max_freq / min_freq) ** torch.linspace(0, 1, dim))


def fourier_embed(t: Tensor, freqs: Tensor) -> Tensor:
    """Sinusoidal embedding for time. Expects input of shape (..., 1)"""
    assert t.size(-1) == 1
    thetas = t.float() * freqs.float()
    return torch.cat([thetas.sin(), thetas.cos()], dim=-1)


class SimpleMLPNet(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.in_proj = FusedLinear(cfg.data_dim, cfg.dim)

        # self.norm_freqs = nn.Buffer(
        #     make_frequencies(cfg.norm_fourier_freqs, cfg.freq_dim)
        # )
        self.time_freqs = nn.Buffer(
            make_frequencies(cfg.time_fourier_freqs, cfg.freq_dim)
        )
        self.diff_freqs = nn.Buffer(
            make_frequencies(cfg.diff_fourier_freqs, cfg.freq_dim)
        )
        # self.emb_proj = FusedLinear(cfg.freq_dim * 6, cfg.dim)
        self.emb_proj = FusedLinear(cfg.freq_dim * 4, cfg.dim)

        block_type = AdaLNMLPBlock if cfg.use_adaln else GatedMLPBlock
        self.blocks = nn.ModuleList(
            [block_type(cfg.dim, cfg.mlp_dim) for _ in range(cfg.depth)]
        )

        self.out_head = Output(cfg.dim, cfg.data_dim, pad_to=64)

    def forward(self, input_ND, r_N1, t_N1, cond_ND):
        x_ND = self.in_proj(input_ND)
        r_N1 = r_N1.expand(x_ND.shape[:-1] + (1,))
        t_N1 = t_N1.expand(x_ND.shape[:-1] + (1,))
        emb = torch.cat(
            [
                # fourier_embed(x_ND.pow(2).mean(dim=-1, keepdim=True), self.norm_freqs),
                fourier_embed(t_N1, self.time_freqs),
                fourier_embed(t_N1 - r_N1, self.diff_freqs),
            ],
            dim=-1,
        )
        emb = emb.type_as(x_ND)
        c_ND = self.emb_proj(emb) + cond_ND
        for block in self.blocks:
            x_ND = block(x_ND, c_ND)
        return self.out_head(x_ND)


class Meanflow(nn.Module):
    def __init__(self, cfg: FlowConfig, net: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.net = net
        self.latent_shape = cfg.sample_shape[:-1] + (2 * cfg.sample_shape[-1],)

    def loss(self, inputs_ND: Tensor, cond_ND: Tensor) -> Tensor:
        N, D = inputs_ND.shape
        device, dtype = inputs_ND.device, inputs_ND.dtype

        mid = torch.sigmoid(
            self.cfg.lognorm_mu
            + self.cfg.lognorm_std * torch.randn((N, 1), device=device)
        )
        diff = torch.rand((N, 1), device=device) * 1.5 / self.cfg.sampling_steps
        r = (mid - diff / 2).clamp(min=0)
        t = (mid + diff / 2).clamp(max=1)

        x = inputs_ND.float()
        e = torch.randn_like(x)

        if self.cfg.schedule == "trig":
            cos_t = torch.cos(torch.pi / 2 * t)
            sin_t = torch.sin(torch.pi / 2 * t)
            xt = cos_t * x + sin_t * e
            vt = (torch.pi / 2) * (-sin_t * x + cos_t * e)
        else:
            assert self.cfg.schedule == "linear"
            xt = (1 - t) * x + t * e
            vt = -x + e

        u, dudt = jvp(
            self.net,
            (xt.type(dtype), r, t, cond_ND),
            (
                vt.type(dtype),
                torch.zeros_like(r),
                torch.ones_like(t),
                torch.zeros_like(cond_ND),
            ),
        )
        u_tgt = vt - (t - r) * dudt.detach()
        loss_distill = (u - u_tgt).pow(2)
        weights = (loss_distill.detach().mean(dim=-1) + 1e-3).pow(
            -self.cfg.loss_wt_power
        )
        loss_distill = loss_distill * weights.unsqueeze(-1)
        loss = loss_distill

        if self.cfg.consistency_steps > 0:
            with torch.no_grad():
                xs = xt
                u_sum = 0
                for i in range(self.cfg.consistency_steps):
                    s1 = t - (t - r) * i / (self.cfg.consistency_steps + 1)
                    s2 = t - (t - r) * (i + 1) / (self.cfg.consistency_steps + 1)
                    us = self.net(xs.type(dtype), s2, s1, cond_ND)
                    xs = xs - (s1 - s2) * us
                    u_sum += us
            loss_consistency = (u - u_sum).pow(2)
            weights = (loss_consistency.detach().mean(dim=-1) + 1e-3).pow(
                -self.cfg.loss_wt_power
            )
            loss_consistency = loss_consistency * weights.unsqueeze(-1)
            metrics.push(
                loss_distill=loss_distill.mean(),
                loss_consistency=loss_consistency.mean(),
            )
            loss = loss + loss_consistency * self.cfg.loss_consistency_wt

        return loss

    def clamp_velocity(self, xt, vt, t, minval: float | None, maxval: float | None):
        if self.cfg.schedule == "linear":
            raise NotImplementedError

        if minval is None and maxval is None:
            return vt
        cos_t = torch.cos(torch.pi / 2 * t)
        sin_t = torch.sin(torch.pi / 2 * t)
        if sin_t < 0.01:
            return vt

        x0 = cos_t * xt - (2 / torch.pi) * sin_t * vt
        x0 = x0.clamp(minval, maxval)
        vt = (torch.pi / 2) * (cos_t * xt - x0) / sin_t
        return vt

    @torch.no_grad()
    def sample(self, cond_input: Tensor) -> Tensor:
        N = cond_input.shape[0]
        xt = torch.randn((N,) + self.cfg.sample_shape, device=cond_input.device)
        timesteps = torch.linspace(1, 0, self.cfg.sampling_steps + 1)
        timesteps = timesteps.to(cond_input.device)[:, None]
        for i in range(self.cfg.sampling_steps):
            ut = self.net(xt, timesteps[i + 1], timesteps[i], cond_input)
            ut = self.clamp_velocity(xt, ut, timesteps[i], -1.0, 1.0)
            xt = xt - (timesteps[i] - timesteps[i + 1]) * ut
        return xt


class MnistFlow(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        net = SimpleMLPNet(cfg.net)
        self.flow = Meanflow(cfg.flow, net)
        self.class_embed = make_embedding(10, cfg.net.dim)

    def forward(self, batch):
        images_NHW, labels_N = batch

        images_N1HW = images_NHW.type(DTYPE) / 255.0
        imgs_ND = images_N1HW.flatten(1, -1)
        imgs_ND = imgs_ND * 2.0 - 1.0

        class_emb_ND = self.class_embed(labels_N).type(DTYPE)

        loss = self.flow.loss(imgs_ND, class_emb_ND).mean()
        metrics.push(loss=loss)
        return loss

    @torch.no_grad()
    def generate(self, labels_N):
        class_emb_ND = self.class_embed(labels_N).type(DTYPE)
        imgs_ND = self.flow.sample(class_emb_ND)
        return imgs_ND


class MnistFlowTrainer(Trainer):
    def on_validation_end(self):
        rows = 5
        # Generate and save samples
        labels_N = torch.arange(10 * rows, device=self.device) // rows
        imgs_ND = self.model.generate(labels_N)
        imgs = rearrange(imgs_ND, "N (H W) -> N H W", H=28, W=28)
        imgs = imgs * 0.5 + 0.5
        imgs = imgs.clamp(0, 1) * 255
        imgs = imgs.type(torch.uint8)
        imgs = imgs.cpu().numpy()

        SAVEDIR.mkdir(exist_ok=True)
        if self.step > 0:
            savepath = SAVEDIR / f"samples_step_{self.step + 1}.png"
        else:
            savepath = SAVEDIR / "samples_initial.png"

        imgs = rearrange(imgs, "(D M) H W -> (M H) (D W)", D=10)
        Image.fromarray(imgs).save(savepath)

        # FID-like metric(s)
        total = 50_000
        samples = []
        counts_histogram = [0 for _ in range(19)]
        for i in tqdm.trange(0, total, self.cfg.micro_batch_size, desc="computing fid"):
            labels_N = (
                torch.arange(i, i + self.cfg.micro_batch_size, device=self.device) % 10
            )
            imgs_ND = self.model.generate(labels_N)
            imgs_ND = imgs_ND * 0.5 + 0.5
            imgs_ND = imgs_ND.clamp(0, 1) * 2.0 - 1.0
            samples.append(imgs_ND)

            n_pixels_on = (imgs_ND > 0.1).long().sum(dim=1) // 16
            for n in n_pixels_on.cpu().numpy():
                counts_histogram[min(n, 18)] += 1

        # Metric based on MLP features
        samples_ND = torch.cat(samples)

        fid_model = FID.from_pretrained()
        fid = fid_model.compute_fid(samples_ND)
        self.metrics.push(mlp_fid=fid)

        # Metric based on distribution of # of pixels enabled per image
        counts_histogram = np.array(counts_histogram)
        counts_histogram = counts_histogram / counts_histogram.sum()
        reference_hist = np.load(Path(__file__).parent / "data/ratios.npy")
        histogram_dist = np.abs(counts_histogram - reference_hist).mean()
        self.metrics.push(hist_fid=histogram_dist)

    def on_epoch_end(self):
        pass


@parse_config
def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)

    model = MnistFlow(cfg)

    mnist_train = MNIST(
        Path(__file__).parent / "data", train=True, download=True, transform=T.ToImage()
    )
    mnist_valid = MNIST(
        Path(__file__).parent / "data",
        train=True,
        download=True,
        transform=T.ToImage(),
    )

    # For MNIST, transfer entire dataset to GPU beforehand for speed.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    train_dataset = TensorDataset(
        mnist_train.data.to(device), mnist_train.targets.to(device)
    )
    valid_dataset = TensorDataset(
        mnist_valid.data.to(device), mnist_valid.targets.to(device)
    )

    try:
        trainer = MnistFlowTrainer(
            cfg.trainer, model, train_dataset, valid_dataset, metrics
        )

        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
