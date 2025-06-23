import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
metrics = Metrics(enabled=False, use_wandb=False)


@dataclass
class FlowConfig:
    patch_size: int = 4
    layer_downfactors: list[int] = field(default_factory=lambda: [1, 4, 1])
    base_dim: int = 384
    mlp_mult: float = 2.0
    cond_dim: int = 512

    use_rotary: bool = True

    t_freq_dim: int = 512
    t_min_freq: float = 2 * torch.pi
    t_max_mult: float = 100

    img_mean: float = 0.5
    img_std: float = 0.5

    noise_ratio: float = 0.1
    sampling_steps: int = 32
    solver: str = "heun"


@dataclass
class Config:
    debug: bool = False
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/vit_flow")

    model: FlowConfig = field(default_factory=FlowConfig)
    do_compile: bool = True
    n_steps: int = 1_000
    valid_every: int | None = 200
    batch_size: int = 2048

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
    def __init__(self, dim: int, mlp_dim: int, cond_dim: int):
        super().__init__()
        self.cond_norm = RMSNorm(cond_dim, affine=False)
        self.cond_proj = FusedLinear(cond_dim, dim, zero_init=True)
        self.norm = RMSNorm(dim, affine=False)
        self.k = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.v = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, c):
        scores = self.k(self.norm(x) * (1 + self.cond_proj(self.cond_norm(c))))
        scores = self.act(scores)
        return self.v(scores)


class Rotary2d(nn.Module):
    def __init__(
        self, npatches: tuple[int, int], head_dim: int, min_freq: float, max_mult: float
    ):
        super().__init__()
        self.nh, self.nw = npatches

        phi = torch.rand(head_dim // 2) * torch.pi * 2
        fh = (
            torch.sin(phi)
            * min_freq
            * (max_mult ** torch.linspace(0, 1, head_dim // 2))
        )
        fw = (
            torch.cos(phi)
            * min_freq
            * (max_mult ** torch.linspace(0, 1, head_dim // 2))
        )

        nh, nw = self.nh, self.nw
        h = torch.linspace(-1, 1, nh).view(nh, 1, 1, 1).expand(nh, nw, 1, 1)
        w = torch.linspace(-1, 1, nw).view(1, nw, 1, 1).expand(nh, nw, 1, 1)
        theta = h * fh + w * fw

        self.cos_HW1d = nn.Buffer(torch.cos(theta), persistent=False)
        self.sin_HW1d = nn.Buffer(torch.sin(theta), persistent=False)

    def forward(self, x_NHWhd):
        """training rotary embedding (known sequence length, uses cached cos/sin)"""
        assert x_NHWhd.shape[1:3] == (self.nh, self.nw)
        x1, x2 = x_NHWhd.float().chunk(2, dim=-1)
        output_NHWhd = torch.cat(
            [
                x1 * self.cos_HW1d - x2 * self.sin_HW1d,
                x1 * self.sin_HW1d + x2 * self.cos_HW1d,
            ],
            dim=-1,
        )
        return output_NHWhd.type_as(x_NHWhd)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        npatches: tuple[int, int],
        downscale: int,
        dim: int,
        mlp_mult: float,
        cond_dim: int,
        head_dim: int,
        rotary_min_freq: float,
        rotary_max_mult: float,
    ):
        super().__init__()
        assert dim % head_dim == 0
        assert npatches[0] % downscale == 0
        assert npatches[1] % downscale == 0
        self.downscale = downscale
        self.nh = npatches[0] // downscale
        self.nw = npatches[1] // downscale
        self.dim = dim
        self.mlp_dim = round(dim * mlp_mult)
        self.head_dim = head_dim
        self.nheads = self.dim // head_dim

        self.norm = RMSNorm(dim, affine=False)
        self.qkv_weight = nn.Parameter(torch.randn(3, dim, dim) / dim**0.5 / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.rotary = Rotary2d(
            (self.nh, self.nw), head_dim, rotary_min_freq, rotary_max_mult
        )
        self.o_proj = FusedLinear(dim, dim, scale=True, zero_init=True)
        self.mlp = ConditionalMLPBlock(dim, self.mlp_dim, cond_dim)

    def forward(self, input_NHWD, cond_ND):
        N = input_NHWD.size(0)
        x_NHWD = rearrange(
            input_NHWD,
            "N (nh dh) (nw dw) (d dd) -> N nh nw (dh dw d) dd",
            dh=self.downscale,
            dw=self.downscale,
            dd=self.downscale,
        ).mean(dim=-1)

        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_NHWD)
        q_NHWhd, k_NHWhd, v_NHWhd = (
            (self.norm(x_NHWD) @ qkv_weight.t())
            .view(N, self.nh, self.nw, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=3)
        )
        q_NHWhd, k_NHWhd = self.qk_norm(q_NHWhd), self.qk_norm(k_NHWhd)
        if self.rotary is not None:
            q_NHWhd, k_NHWhd = self.rotary(q_NHWhd), self.rotary(k_NHWhd)

        x_NhTd = F.scaled_dot_product_attention(
            rearrange(q_NHWhd, "N H W h d -> N h (H W) d", H=self.nh, W=self.nw),
            rearrange(k_NHWhd, "N H W h d -> N h (H W) d", H=self.nh, W=self.nw),
            rearrange(v_NHWhd, "N H W h d -> N h (H W) d", H=self.nh, W=self.nw),
            is_causal=False,
        )
        x_NHWD = rearrange(x_NhTd, "N h (H W) d -> N H W (h d)", H=self.nh, W=self.nw)
        x_NHWD = self.o_proj(x_NHWD)
        x_NHWD = x_NHWD + self.mlp(x_NHWD, cond_ND.unsqueeze(1).unsqueeze(2))

        x_NHWD = einops.repeat(
            x_NHWD,
            "N nh nw (rh rw d) -> N (nh rh) (nw rw) (d dd)",
            rh=self.downscale,
            rw=self.downscale,
            dd=self.downscale,
        )
        x_NHWD = input_NHWD + x_NHWD

        return x_NHWD


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        layer_downfactors: list[int],
        base_dim: int,
        mlp_mult: float,
        cond_dim: int,
        head_dim: int,
        use_rotary: bool,
        rotary_min_freq: float = torch.pi / 2,
        rotary_max_mult: float = 32.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.nh = image_size[0] // patch_size[0]
        self.nw = image_size[1] // patch_size[1]

        self.patchify_wt = nn.Parameter(
            torch.randn(base_dim, patch_size[0] * patch_size[1])
            / (2 * (patch_size[0] * patch_size[1]) ** 0.5)
        )
        if use_rotary:
            self.pos_emb = None
        else:
            self.pos_emb = nn.Parameter(torch.randn(self.nh, self.nw, base_dim) * 0.05)
            self.pos_emb._is_embed = True
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    (self.nh, self.nw),
                    down,
                    base_dim,
                    mlp_mult,
                    cond_dim,
                    head_dim,
                    rotary_min_freq,
                    rotary_max_mult,
                )
                for down in layer_downfactors
            ]
        )
        self.out_head = nn.Sequential(
            RMSNorm(base_dim, affine=False),
            FusedLinear(base_dim, patch_size[0] * patch_size[1]),
        )
        self.out_head[-1].weight._is_output = True

    def forward(self, img_NHW, cond_ND):
        ph = self.patch_size[0]
        pw = self.patch_size[1]
        x_NHWD = rearrange(
            img_NHW, "N (nh ph) (nw pw) -> N nh nw (ph pw)", ph=ph, pw=pw
        )
        x_NHWD = x_NHWD @ self.patchify_wt.type_as(x_NHWD).t()
        if self.pos_emb is not None:
            x_NHWD = x_NHWD + self.pos_emb

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
        assert self.cfg.solver in ("heun", "analytic")

        self.t_proj = FusedLinear(cfg.t_freq_dim * 2, cfg.cond_dim)
        self.class_embed = make_embedding(10, cfg.cond_dim)
        self.vit = VisionTransformer(
            (32, 32),
            (cfg.patch_size, cfg.patch_size),
            layer_downfactors=cfg.layer_downfactors,
            base_dim=cfg.base_dim,
            mlp_mult=cfg.mlp_mult,
            cond_dim=cfg.cond_dim,
            head_dim=64,
            use_rotary=cfg.use_rotary,
        )
        self.freqs = nn.Buffer(
            self.cfg.t_min_freq
            * (self.cfg.t_max_mult ** torch.linspace(0, 1, self.cfg.t_freq_dim))
        )

    def time_embed(self, t):
        assert t.size(-1) == 1
        assert t.amin() >= 0.0, f"{t.amin()=:.7f}"
        assert t.amax() <= 1.0, f"{t.amax()=:.7f}"
        thetas = t.float() * self.freqs.float()
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

        pt = self(xt.type(DTYPE), t, digits).flatten(1, -1)

        loss = (pt - vt).pow(2).mean()
        return loss

    def forward(self, xt_ND, t_N1, digits_N):
        N = xt_ND.size(0)
        xt_NHW = rearrange(xt_ND, "N (H W) -> N H W", H=28, W=28)
        xt_NHW = F.pad(xt_NHW, (2, 2, 2, 2))
        assert xt_NHW.shape == (N, 32, 32)
        t_emb_ND = self.t_proj(self.time_embed(t_N1).type(DTYPE))
        cond_ND = self.class_embed(digits_N).type(DTYPE)
        output = self.vit(xt_NHW, t_emb_ND + cond_ND).float()
        assert output.shape == (N, 32, 32)
        output = output[..., 2:-2, 2:-2]
        return output

    def clamp_velocity(self, xt, vt, s, minval: float | None, maxval: float | None):
        if minval is None and maxval is None:
            return vt
        if torch.cos(s) < 0.01:
            return vt

        x1 = torch.sin(s) * xt + (2 / torch.pi) * torch.cos(s) * vt
        x1 = torch.clamp(x1, minval, maxval)
        vt = (torch.pi / 2) * (x1 - torch.sin(s) * xt) / torch.cos(s)
        return vt

    @torch.no_grad()
    def sample(
        self,
        c: Tensor,
        noise_ratio: float,
        n_steps: int,
        u: Tensor | None = None,
        lam: float = 0.0,
        minval: float | None = None,
        maxval: float | None = None,
        analytic_step: bool = True,
        use_heun: bool = False,
    ) -> Tensor:
        """
        Args:
            c: conditioning input
            noise_ratio: ratio of the amount of noise added to the progress per step.
                In each step, we take a reverse step (from ta -> tc, removing noise),
                then a forward step (from tc -> tb, adding noise back), akin to Langevin
                dynamics. Set noise_ratio to 0 for deterministic sampling.
            n_steps: total # of model evaluations
            u: unconditional input (for classifier-free guidance)
            lam: classifier-free guidance scale
            minval: force samples to be above this value
            maxval: force samples to be below this value
            analytic_step: when True, the sampler leverages the fact that we know xt =
                sin(s) x + cos(s) z and vt = pi/2 cos(s) x - pi/2 sin(s) z, so the
                value of xt' for the next t' can be estimated by solving for x and z
            use_heun: when True, two velocity estimates are averaged for each step
        """
        device = c.device

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt = torch.randn(c.shape + (28 * 28,), device=device)
        noise_dt = noise_ratio / n_steps
        do_guidance = (u is not None) and (lam != 0)

        for i in range(n_steps):
            ta = timesteps[i].view(1)
            tb = timesteps[i + 1].view(1)
            tc = (timesteps[i + 1].view(1) + noise_dt).clamp(max=1)
            sa = ta * (torch.pi / 2)
            sb = tb * (torch.pi / 2)
            sc = tc * (torch.pi / 2)

            va = self(xt.type(DTYPE), ta, c).flatten(-2, -1)
            if do_guidance:
                va_uncond = self(xt.type(DTYPE), ta, u).flatten(-2, -1)
                va = va * (lam + 1.0) - va_uncond * lam
            va = self.clamp_velocity(xt, va, sa, minval, maxval)

            if analytic_step:
                xc = torch.cos(sa - sc) * xt - (2 / torch.pi) * torch.sin(sa - sc) * va
            else:
                xc = xt + (tc - ta) * va

            if use_heun:
                vc = self(xc.type(DTYPE), tc, c).flatten(-2, -1)
                if do_guidance:
                    vc_uncond = self(vc.type(DTYPE), tc, u).flatten(-2, -1)
                    vc = vc * (lam + 1.0) - vc_uncond * lam
                vc = self.clamp_velocity(xc, vc, sc, minval, maxval)

                if analytic_step:
                    # Using heun with analytic_step=True is a bit sus.
                    xc = torch.cos(sa - sc) * xt - (1 / torch.pi) * torch.sin(
                        sa - sc
                    ) * (va + vc)
                else:
                    xc = xt + 0.5 * (tc - ta) * (va + vc)

            if sb < sc:
                scale = sb.sin() / sc.sin()
                extra_var = sb.cos().pow(2) - (sc.cos() * scale).pow(2)
                xt = scale * xc + extra_var.sqrt() * torch.randn_like(xt)
            else:
                xt = xc
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

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()
        metrics.context = "valid_"
        for images, labels in mnist_loader(
            train=False, batch_size=self.cfg.batch_size, epochs=1, device=self.device
        ):
            self.model.loss(images, labels)

        rows = 5
        # Generate and save samples
        kwargs = dict(
            noise_ratio=self.cfg.model.noise_ratio,
            n_steps=self.cfg.model.sampling_steps,
            minval=-self.cfg.model.img_mean / self.cfg.model.img_std,
            maxval=(1 - self.cfg.model.img_mean) / self.cfg.model.img_std,
            analytic_step=self.cfg.model.solver == "analytic",
            use_heun=self.cfg.model.solver == "heun",
        )
        labels_N = torch.arange(10 * rows, device=self.device) // rows
        imgs_ND = self.model.sample(labels_N, **kwargs)
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

        # FID-like metric(s)
        total = 50_000
        samples = []
        counts_histogram = [0 for _ in range(19)]
        for i in tqdm.trange(0, total, self.cfg.batch_size, desc="computing fid"):
            labels_N = torch.arange(i, i + self.cfg.batch_size, device=self.device) % 10
            imgs_ND = self.model.sample(labels_N, **kwargs)
            imgs_ND = imgs_ND * self.model.cfg.img_std + self.model.cfg.img_mean
            imgs_ND = imgs_ND.clamp(0, 1) * 2.0 - 1.0
            samples.append(imgs_ND)

            n_pixels_on = (imgs_ND > 0.1).long().sum(dim=1) // 16
            for n in n_pixels_on.cpu().numpy():
                counts_histogram[min(n, 18)] += 1

        # Metric based on MLP features
        samples_ND = torch.cat(samples)
        fid = self.fid.compute_fid(samples_ND)
        metrics.push(mlp_fid=fid)

        # Metric based on distribution of # of pixels enabled per image
        counts_histogram = np.array(counts_histogram)
        counts_histogram = counts_histogram / counts_histogram.sum()
        reference_hist = np.load(Path(__file__).parent / "data/ratios.npy")
        histogram_dist = np.abs(counts_histogram - reference_hist).mean()
        metrics.push(hist_fid=histogram_dist)

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


# TODO: update to the other version...
@parse_config
def main(cfg: Config):
    logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)

    try:
        trainer = Trainer(cfg)
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
