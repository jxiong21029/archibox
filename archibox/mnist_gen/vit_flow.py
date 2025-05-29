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

from archibox.components import FusedLinear, RMSNorm
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32
metrics = Metrics(enabled=False, use_wandb=False)


@dataclass
class FlowConfig:
    patch_size: int = 8
    dim: int = 512
    mlp_dim: int = 1024
    depth: int = 2

    freq_dim: int = 1024
    min_freq: float = 2 * torch.pi
    max_freq: float = 200 * torch.pi

    img_mean: float = 0.5
    img_std: float = 0.5

    churn_ratio: float = 0.2
    sampling_steps: int = 512


@dataclass
class Config:
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/vit_flow")

    model: FlowConfig = field(default_factory=FlowConfig)
    do_compile: bool = True
    n_steps: int = 1_000
    valid_every: int | None = 200
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
        self.wc = FusedLinear(dim, dim, zero_init=True)
        self.k = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.v = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, c):
        scores = self.k(self.norm(x) * (1 + self.wc(self.norm(c))))
        scores = self.act(scores)
        return self.v(scores)


class Rotary2d(nn.Module):
    def __init__(self, npatches: tuple[int, int], head_dim: int, base: float):
        super().__init__()
        self.nh, self.nw = npatches
        freqs = (1 / base) ** torch.linspace(0, 1, head_dim // 4, dtype=torch.float32)
        # Used during inference
        self.freqs_d = nn.Buffer(freqs, persistent=False)
        # Used during training
        theta_HW1d = torch.cat(
            [
                torch.arange(self.nh).view(self.nh, 1, 1).expand(self.nh, self.nw, 1)
                * freqs,
                torch.arange(self.nw).view(1, self.nw, 1).expand(self.nh, self.nw, 1)
                * freqs,
            ],
            dim=-1,
        ).unsqueeze(-2)
        assert theta_HW1d.shape == (self.nh, self.nw, 1, head_dim // 2)
        self.cos_HW1d = nn.Buffer(torch.cos(theta_HW1d), persistent=False)
        self.sin_HW1d = nn.Buffer(torch.sin(theta_HW1d), persistent=False)

    def forward(self, x_NHWhd):
        """training rotary embedding (known sequence length, uses cached cos/sin)"""
        assert x_NHWhd.shape[1:3] == (self.nh, self.nw)
        x1_NHWhd, x2_NHWhd = x_NHWhd.float().chunk(2, dim=-1)
        o1_NHWhd = x1_NHWhd * self.cos_HW1d + x2_NHWhd * self.sin_HW1d
        o2_NHWhd = x1_NHWhd * (-self.sin_HW1d) + x2_NHWhd * self.cos_HW1d
        return torch.cat([o1_NHWhd, o2_NHWhd], dim=-1).type_as(x_NHWhd)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        npatches: tuple[int, int],
        dim: int,
        mlp_dim: int,
        head_dim: int,
        rotary: Rotary2d | None,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.nh, self.nw = npatches
        self.dim = dim
        self.head_dim = head_dim
        self.nheads = dim // head_dim
        self.rotary = rotary

        self.norm = RMSNorm(dim, affine=False)
        self.qkv_weight = nn.Parameter(torch.randn(3, dim, dim) / dim**0.5 / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.o_proj = FusedLinear(dim, dim, scale=True, zero_init=True)
        self.mlp = ConditionalMLPBlock(dim, mlp_dim)

    def forward(self, input_NHWD, cond_ND):
        N, H, W, _ = input_NHWD.shape
        assert H == self.nh and W == self.nw
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_NHWD)
        q_NHWhd, k_NHWhd, v_NHWhd = (
            (self.norm(input_NHWD) @ qkv_weight.t())
            .view(N, H, W, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=3)
        )
        q_NHWhd, k_NHWhd = self.qk_norm(q_NHWhd), self.qk_norm(k_NHWhd)
        if self.rotary is not None:
            q_NHWhd, k_NHWhd = self.rotary(q_NHWhd), self.rotary(k_NHWhd)

        x_NhTd = F.scaled_dot_product_attention(
            rearrange(q_NHWhd, "N H W h d -> N h (H W) d", H=H, W=W),
            rearrange(k_NHWhd, "N H W h d -> N h (H W) d", H=H, W=W),
            rearrange(v_NHWhd, "N H W h d -> N h (H W) d", H=H, W=W),
            is_causal=False,
        )
        x_NHWD = rearrange(x_NhTd, "N h (H W) d -> N H W (h d)", H=H, W=W)
        x_NHWD = input_NHWD + self.o_proj(x_NHWD)

        x_NHWD = x_NHWD + self.mlp(x_NHWD, cond_ND.unsqueeze(1).unsqueeze(2))
        return x_NHWD


class ViT(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        dim: int,
        mlp_dim: int,
        head_dim: int,
        depth: int,
        rope_base: float,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.nh = image_size[0] // patch_size[0]
        self.nw = image_size[1] // patch_size[1]

        self.patchify_wt = nn.Parameter(
            torch.randn(dim, patch_size[0] * patch_size[1])
            / (2 * (patch_size[0] * patch_size[1]) ** 0.5)
        )
        self.rotary = Rotary2d((self.nh, self.nw), head_dim, rope_base)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock((self.nh, self.nw), dim, mlp_dim, head_dim, self.rotary)
                for _ in range(depth)
            ]
        )
        self.out_head = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, patch_size[0] * patch_size[1]),
        )
        self.out_head[-1].weight._is_output = True

    def forward(self, img_NHW, cond_ND):
        ph = self.patch_size[0]
        pw = self.patch_size[1]
        x_NHWD = rearrange(
            img_NHW, "N (nh ph) (nw pw) -> N nh nw (ph pw)", ph=ph, pw=pw
        )
        x_NHWD = x_NHWD @ self.patchify_wt.type_as(x_NHWD).t()

        for block in self.blocks:
            x_NHWD = block(x_NHWD, cond_ND)

        x_NHWD = self.out_head(x_NHWD)
        return rearrange(
            x_NHWD,
            "N nh nw (ph pw) -> N (nh ph) (nw pw)",
            nh=self.nh,
            nw=self.nw,
            ph=ph,
            pw=pw,
        )


class MnistFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.t_proj = FusedLinear(cfg.freq_dim * 2, cfg.dim)
        self.class_embed = make_embedding(10, cfg.dim)
        self.vit = ViT(
            (32, 32),
            (cfg.patch_size, cfg.patch_size),
            dim=cfg.dim,
            mlp_dim=cfg.mlp_dim,
            head_dim=64,
            depth=cfg.depth,
            rope_base=32.0,
        )

    def time_embed(self, t):
        assert t.size(-1) == 1
        assert t.amin() >= 0.0, f"{t.amin()=:.7f}"
        assert t.amax() <= 1.0, f"{t.amax()=:.7f}"
        freqs = self.cfg.min_freq * (
            (self.cfg.max_freq / self.cfg.min_freq)
            ** torch.linspace(0, 1, self.cfg.freq_dim, device=t.device)
        )
        thetas = t.float() * freqs.float()
        return torch.cat([thetas.sin(), thetas.cos()], dim=-1)

    def loss(self, inputs_NHW, digits):
        N, H, W = inputs_NHW.shape
        device = inputs_NHW.device

        inputs_NHW = inputs_NHW.float() / 255.0
        inputs_NHW = (inputs_NHW.type(DTYPE) - self.cfg.img_mean) / self.cfg.img_std

        x1 = inputs_NHW.flatten(1, -1)
        t = torch.sigmoid(torch.randn((inputs_NHW.size(0), 1), device=device))

        x0 = torch.randn_like(x1)
        s = t * (torch.pi / 2)
        xt = torch.cos(s) * x0 + torch.sin(s) * x1
        vt = (-torch.sin(s) * x0 + torch.cos(s) * x1) * (torch.pi / 2)

        pt = self(xt.type(DTYPE).view(N, H, W), t, digits).flatten(1, -1)

        loss = (pt - vt).pow(2).mean()
        return loss

    def forward(self, xt_NHW, t_N1, digits_N):
        N = xt_NHW.size(0)
        assert xt_NHW.shape[1:] == (28, 28)

        xt_NHW = F.pad(xt_NHW, (2, 2, 2, 2))
        assert xt_NHW.shape == (N, 32, 32)
        t_emb_ND = self.t_proj(self.time_embed(t_N1).type(DTYPE))
        cond_ND = self.class_embed(digits_N).type(DTYPE)
        output = self.vit(xt_NHW, t_emb_ND + cond_ND).float()
        assert output.shape == (N, 32, 32)
        output = output[..., 2:-2, 2:-2]
        return output

    @torch.no_grad
    def sample_heun(self, digits, n_steps: int = 50):
        N = digits.size(0)
        device = digits.device

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt = torch.randn((N, 28 * 28), device=device)

        for i in range(n_steps):
            ta = timesteps[i].view(1)
            tb = timesteps[i + 1].view(1)

            va = self(xt.type(DTYPE).view(N, 28, 28), ta, digits).flatten(1, -1)
            xb = xt + (tb - ta) * va
            vb = self(xb.type(DTYPE).view(N, 28, 28), tb, digits).flatten(1, -1)

            xt = xt + (tb - ta) * (va + vb) / 2
        return xt

    @torch.no_grad
    def sample_analytic(self, digits, churn_ratio: float, n_steps: int = 50):
        N = digits.size(0)
        device = digits.device

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt = torch.randn((N, 28 * 28), device=device)
        churn_dt = churn_ratio / n_steps

        for i in range(n_steps):
            ta = timesteps[i].view(1)
            tb = timesteps[i + 1].view(1)
            tA = (ta - churn_dt).clamp(min=0)
            sa = ta * (torch.pi / 2)
            sb = tb * (torch.pi / 2)
            sA = tA * (torch.pi / 2)

            if tA < ta:
                scale = sA.sin() / sa.sin()
                extra_var = sA.cos().pow(2) - (sa.cos() * scale).pow(2)
                xA = scale * xt + extra_var.sqrt() * torch.randn_like(xt)
            else:
                xA = xt

            vA = self(xA.type(DTYPE).view(N, 28, 28), tA, digits).flatten(1, -1)
            xt = torch.cos(sA - sb) * xA - (2 / torch.pi) * torch.sin(sA - sb) * vA
        return xt


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

        metrics.enabled = self.is_main_process
        metrics.use_wandb = cfg.use_wandb
        metrics.context = "train_"

        dataset_path = Path(__file__).parent / "data"
        dataset_path.mkdir(exist_ok=True)

        self.train_loader = mnist_loader(
            train=True, batch_size=cfg.batch_size, device=self.device
        )
        self.model = MnistFlow(cfg.model).to(self.device)
        if cfg.do_compile:
            self.model = torch.compile(self.model, mode="max-autotune")

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
        loss = self.model.loss(images, labels)
        loss.backward()
        for optim in self.optims:
            optim.step()
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

    @torch.no_grad
    def valid_epoch(self):
        self.model.eval()
        metrics.context = "valid_"
        for images, labels in mnist_loader(
            train=False, batch_size=self.cfg.batch_size, epochs=1, device=self.device
        ):
            self.model.loss(images, labels)

        rows = 5
        # Generate and save samples
        labels_N = torch.arange(10 * rows, device=self.device) // rows
        imgs_ND = self.model.sample_analytic(
            labels_N,
            churn_ratio=self.model.cfg.churn_ratio,
            n_steps=self.model.cfg.sampling_steps,
        )
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

        # Compute FID-like metric based on distribution of # pixels on per image
        total = 10_000
        counts = [0 for _ in range(19)]
        for i in range(0, total, self.cfg.batch_size):
            labels_N = torch.arange(i, i + self.cfg.batch_size, device=self.device) % 10
            imgs_ND = self.model.sample_analytic(
                labels_N,
                churn_ratio=self.model.cfg.churn_ratio,
                n_steps=self.model.cfg.sampling_steps,
            )
            imgs_ND = imgs_ND * self.model.cfg.img_std + self.model.cfg.img_mean
            num_on_N = (imgs_ND.clamp(0, 1) > 0.1).long().sum(dim=1)
            num_on_N = num_on_N // 16
            for n in num_on_N.cpu().numpy():
                counts[min(n, 18)] += 1
        counts = np.array(counts)
        fracs = counts / counts.sum()
        ref_fracs = np.load(Path(__file__).parent / "data/ratios.npy")
        distance = np.abs(fracs - ref_fracs).mean()
        metrics.push(pixel_counts_distance=distance)
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--churn_ratio", type=float, default=3.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    cfg = Config(model=FlowConfig(churn_ratio=args.churn_ratio))
    try:
        trainer = Trainer(cfg)
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
