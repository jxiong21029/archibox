import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from archibox.components import FusedLinear, RMSNorm
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon
from einops import rearrange
from PIL import Image

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


@dataclass
class FlowConfig:
    dim: int = 768
    mlp_dim: int = 3072
    embed_dim: int = 128
    cond_dim: int = 256
    depth: int = 2

    img_mean: float = 0.1307
    img_std: float = 0.3081
    use_trigflow: bool = False


@dataclass
class Config:
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/flow_v2")

    model: FlowConfig = field(default_factory=FlowConfig)
    do_compile: bool = False
    n_steps: int = 5_000
    valid_every: int | None = 1000
    batch_size: int = 8192

    muon_lr: float = 0.02
    muon_mu: float = 0.9
    muon_wd: float = 0.01
    scalar_lr: float = 0.001
    embeds_lr: float = 0.001
    output_lr: float = 0.001
    adamw_mu1: float = 0.9
    adamw_mu2: float = 0.99
    adamw_wd: float = 0.01
    lr_cooldown_start: int | None = None
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
        scores = self.act(scores * (1 + self.wc(c)))
        return x + self.wv(scores)


class VectorFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        data_dim: int,
        cond_dim: int,
        depth: int,
        use_trigflow: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.use_trigflow = use_trigflow

        self.in_proj = FusedLinear(data_dim, dim)
        self.t_proj = FusedLinear(cond_dim, cond_dim)
        self.blocks = nn.ModuleList(
            [ModulatedMLP(dim, mlp_dim, cond_dim=cond_dim) for _ in range(depth)]
        )
        self.out_head = nn.Sequential(
            RMSNorm(dim, affine=False), FusedLinear(dim, data_dim, zero_init=True)
        )
        self.out_head[-1].weight._is_output = True

    def time_embed(self, t, freq_lo=1.0, freq_hi=16.0):
        assert t.size(-1) == 1
        freqs_D = freq_lo * (
            freq_hi ** torch.linspace(0, 1, self.cond_dim // 2, device=t.device).float()
        )
        theta_ND = t.float() * freqs_D
        return torch.cat([theta_ND.sin(), theta_ND.cos()], dim=-1)

    def forward(self, inputs, cond):
        assert inputs.size(-1) == self.data_dim
        assert cond.size(-1) == self.cond_dim
        device, dtype = inputs.device, inputs.dtype

        x1 = inputs.float()
        t = torch.sigmoid(torch.randn(inputs.shape[:-1] + (1,), device=device))
        # t = torch.rand(inputs.shape[:-1] + (1,), device=device)
        t_emb = self.t_proj(self.time_embed(t).type(dtype))

        x0 = torch.randn_like(x1)
        if self.use_trigflow:
            theta = t * (torch.pi / 2)
            xt = torch.cos(theta) * x0 + torch.sin(theta) * x1
            vt = (torch.cos(theta) * x1 - torch.sin(theta) * x0) * (torch.pi / 2)
        else:
            xt = (1 - t) * x0 + t * x1
            vt = x1 - x0

        h = self.in_proj(xt.type(dtype))
        for block in self.blocks:
            h = block(h, cond + t_emb)
        pt = self.out_head(h).type_as(vt)

        loss = (pt - vt).pow(2)
        return loss

    def step(self, xt, c):
        h = self.in_proj(xt)
        for block in self.blocks:
            h = block(h, c)
        return self.out_head(h).float()

    @torch.no_grad
    def generate(self, cond, n_steps: int = 50):
        device, dtype = cond.device, cond.dtype

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        xt = torch.randn(cond.shape[:-1] + (self.data_dim,), device=device)

        for i in range(n_steps):
            t_start = timesteps[i].view(1)
            t_stop = timesteps[i + 1].view(1)
            t_start_emb = self.t_proj(self.time_embed(t_start).type(dtype))
            t_stop_emb = self.t_proj(self.time_embed(t_stop).type(dtype))

            v_start = self.step(xt.type(dtype), cond + t_start_emb)
            xt_stop = xt + (t_stop - t_start) * v_start
            v_stop = self.step(xt_stop.type(dtype), cond + t_stop_emb)

            xt = xt + (t_stop - t_start) * (v_start + v_stop) / 2
        return xt


class MnistFlow(nn.Module):
    def __init__(self, cfg: FlowConfig, metrics: Metrics):
        super().__init__()
        self.cfg = cfg
        self.metrics = metrics
        self.flow = VectorFlow(
            cfg.dim,
            cfg.mlp_dim,
            data_dim=28 * 28,
            cond_dim=cfg.cond_dim,
            depth=cfg.depth,
            use_trigflow=cfg.use_trigflow,
        )
        self.class_embed = make_embedding(10, cfg.cond_dim)

    def forward(self, images_NHW, labels_N):
        images_N1HW = images_NHW.type(DTYPE) / 255.0
        imgs_ND = images_N1HW.flatten(1, -1)
        imgs_ND = (imgs_ND - self.cfg.img_mean) / self.cfg.img_std

        class_emb_ND = self.class_embed(labels_N).type(DTYPE)

        loss = self.flow(imgs_ND, class_emb_ND).mean()
        self.metrics.push(loss=loss)
        return loss

    @torch.no_grad
    def generate(self, labels_N, n_steps: int = 50):
        class_emb_ND = self.class_embed(labels_N).type(DTYPE)
        imgs_ND = self.flow.generate(class_emb_ND, n_steps=n_steps)
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

        # Compute FID-like metric based on distribution of # pixels on per image
        total = 10_000
        counts = [0 for _ in range(19)]
        for i in range(0, total, self.cfg.batch_size):
            labels_N = torch.arange(i, i + self.cfg.batch_size, device=self.device) % 10
            imgs_ND = self.model.generate(labels_N)
            imgs_ND = imgs_ND * self.model.cfg.img_std + self.model.cfg.img_mean
            num_on_N = (imgs_ND.clamp(0, 1) > 0.1).long().sum(dim=1)
            num_on_N = num_on_N // 16
            for n in num_on_N.cpu().numpy():
                counts[min(n, 18)] += 1
        counts = np.array(counts)
        fracs = counts / counts.sum()
        ref_fracs = np.load(Path(__file__).parent / "data/ratios.npy")
        distance = np.abs(fracs - ref_fracs).mean()
        self.metrics.push(pixel_counts_distance=distance)

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
