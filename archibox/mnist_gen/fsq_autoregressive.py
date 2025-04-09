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
from torch import Tensor
from torch.distributions import Categorical

from archibox.components import Decoder, FusedLinear, ReLU2, RMSNorm
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon

"""
TODO variants:
- FSQ (chunked) autoregressive
- FSQ (pure/corrective) masked diffusion
"""

log = logging.getLogger(__name__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


IMAGE_MEAN = 0.1307
IMAGE_STD = 0.3081


@dataclass
class MnistFSQConfig:
    dim: int = 384
    mlp_dim: int = 768
    encoder_depth: int = 3
    decoder_depth: int = 6
    predictor_depth: int = 6
    latent_bins: int = 4
    latent_ndim: int = 448
    latent_groups: int = 64


@dataclass
class Config:
    use_wandb: bool = False

    model: MnistFSQConfig = field(default_factory=MnistFSQConfig)
    do_compile: bool = True
    n_steps: int = 10_000
    valid_every: int | None = 500
    batch_size: int = 2048

    muon_lr: float = 0.04
    muon_mu: float = 0.9
    muon_wd: float = 0.01
    scalar_lr: float = 0.002
    embeds_lr: float = 0.002
    output_lr: float = 0.002
    adamw_mu1: float = 0.9
    adamw_mu2: float = 0.99
    adamw_wd: float = 0.01
    lr_cooldown_start: int | None = 5000
    lr_cooldown_ratio: float = 0.0


def make_embedding(num_embeddings: int, embedding_dim: int):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    embed.to(dtype=DTYPE)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


class FSQ(nn.Module):
    def __init__(self, dim_in: int, latent_ndim: int, latent_bins: int):
        super().__init__()
        self.latent_bins = latent_bins
        self.in_proj = nn.Sequential(
            RMSNorm(dim_in, affine=False),
            FusedLinear(dim_in, latent_ndim),
        )

    def forward(self, x: Tensor):
        z = torch.tanh(self.in_proj(x).float())  # in [-1, 1]
        z = (z + 1.0) * ((self.latent_bins - 1) / 2.0)  # [-1, 1] --> [0, nbins - 1]
        idxs = torch.round(z.detach())
        z = z + (idxs - z).detach()  # round w/ straight-through grad
        z = z * (2.0 / (self.latent_bins - 1)) - 1.0  # [0, nbins - 1] --> [-1, 1]
        return z.type_as(x), idxs.type(torch.long)


class MLPStack(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                RMSNorm(dim, affine=False),
                FusedLinear(dim, mlp_dim),
                ReLU2(),
                FusedLinear(mlp_dim, dim, zero_init=True),
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class FSQDecoder(nn.Module):
    """Autoregressively models latents"""

    def __init__(self, dim, mlp_dim, depth, latent_ndim, latent_bins, latent_groups):
        super().__init__()
        assert latent_ndim % latent_groups == 0
        self.latent_bins = latent_bins
        self.seq_len = latent_groups
        self.g_size = latent_ndim // latent_groups
        self.vocab_size = latent_bins**self.g_size
        log.info(f"model vocab_size={self.vocab_size}")

        self.input_emb = make_embedding(self.vocab_size + 1, dim)
        self.pos_emb = make_embedding(self.seq_len, dim)
        self.decoder = Decoder(
            dim=dim,
            mlp_dim=mlp_dim,
            depth=depth,
            seq_len=self.seq_len,
            window_size=self.seq_len,
            head_dim=64,
            use_rope=False,
            scale=False,
        )
        self.out_head = FusedLinear(dim, self.vocab_size, zero_init=True)
        self.out_head.weight._is_output = True

    def forward(self, ids_NL, c_ND):
        ids_NLG = rearrange(ids_NL, "N (L G) -> N L G", L=self.seq_len, G=self.g_size)
        group_ids_NL = torch.sum(
            ids_NLG
            * (self.latent_bins ** torch.arange(self.g_size, device=ids_NL.device)),
            dim=-1,
        )
        input_ids_NL = torch.cat(
            [torch.zeros_like(group_ids_NL[:, :1]), 1 + group_ids_NL[:, :-1]], dim=1
        )
        x_NLD = self.input_emb(input_ids_NL) + c_ND.unsqueeze(1)
        x_NLD = x_NLD + self.pos_emb.weight
        x_NLD = self.decoder(x_NLD)
        logits_NLD = self.out_head(x_NLD).float()
        return F.cross_entropy(logits_NLD.flatten(0, -2), group_ids_NL.flatten())

    def generate(self, c_ND):
        N, _ = c_ND.shape
        device = c_ND.device
        input_ids_N = torch.zeros(N, device=device, dtype=torch.long)
        kv_cache = None
        output_ids = []
        for t in range(self.seq_len):
            x_ND = self.input_emb(input_ids_N) + c_ND + self.pos_emb.weight[t]
            x_ND, kv_cache = self.decoder.predict(x_ND, kv_cache, t=None)
            output_ids_N = Categorical(logits=self.out_head(x_ND)).sample()
            output_ids.append(output_ids_N)
            input_ids_N = output_ids_N + 1
        output_ids_NL = torch.stack(output_ids, dim=1)

        ids_NLG = torch.zeros(
            N, self.seq_len, self.g_size, device=device, dtype=torch.long
        )
        for g in range(self.g_size):
            ids_NLG[:, :, g] = output_ids_NL % self.latent_bins
            output_ids_NL = output_ids_NL // self.latent_bins
        return ids_NLG.flatten(1, -1)


class CondFSQ(nn.Module):
    def __init__(self, cfg, metrics):
        super().__init__()
        self.cfg = cfg
        self.metrics = metrics
        self.class_embed = make_embedding(10, cfg.dim)
        self.encoder_proj = FusedLinear(28 * 28, cfg.dim)
        self.encoder = MLPStack(cfg.dim, cfg.mlp_dim, cfg.encoder_depth)

        self.fsq = FSQ(cfg.dim, cfg.latent_ndim, cfg.latent_bins)

        self.decoder = FSQDecoder(
            cfg.dim,
            cfg.mlp_dim,
            cfg.decoder_depth,
            cfg.latent_ndim,
            cfg.latent_bins,
            cfg.latent_groups,
        )
        self.predictor_proj = FusedLinear(cfg.latent_ndim, cfg.dim)
        self.predictor = MLPStack(cfg.dim, cfg.mlp_dim, cfg.predictor_depth)
        self.predictor_out = FusedLinear(cfg.dim, 28 * 28, zero_init=True)
        self.predictor_out.weight._is_output = True

    def forward(self, images_NHW, labels_N):
        N, _, _ = images_NHW.shape
        assert images_NHW.dtype == torch.uint8
        assert labels_N.shape == (N,)
        images_N1HW = images_NHW.type(DTYPE) / 255.0
        imgs_ND = images_N1HW.flatten(1, -1)
        imgs_ND = (imgs_ND - IMAGE_MEAN) / IMAGE_STD
        x_ND = self.encoder_proj(imgs_ND)
        c_ND = self.class_embed(labels_N)
        x_ND = self.encoder(x_ND + c_ND)
        z_NL, ids_NL = self.fsq(x_ND)
        decoder_xent = self.decoder(ids_NL, c_ND)

        x_ND = self.predictor_proj(z_NL)
        x_ND = self.predictor(x_ND + c_ND)
        p_ND = self.predictor_out(x_ND)
        predictor_mse = (p_ND - imgs_ND).pow(2).mean()

        # mean_regularizer = z_NL.mean(dim=0).pow(2).mean()
        # feature_covar_LL = (z_NL.t() @ z_NL) / N
        # covar_regularizer = (
        #     feature_covar_LL.pow(2).sum() - torch.diag(feature_covar_LL).pow(2).sum()
        # )
        # self.metrics.push(
        #     mean_regularizer=mean_regularizer, covar_regularizer=covar_regularizer
        # )

        loss = decoder_xent + predictor_mse
        # loss = loss + 0.1 * mean_regularizer + 0.1 * covar_regularizer

        self.metrics.push(
            loss=loss,
            decoder_xent=decoder_xent,
            predictor_mse=predictor_mse,
            latent_std=z_NL.std(dim=0).mean(),
            prediction_std=p_ND.std(),
        )
        return loss

        # Should the latent be conditional? Yes.
        # input + cond --(FSQ)-> latents
        # cond + latents[:i] --(xent)-> latents[i]
        # cond + latents --(MSE)-> reconstruction

    @torch.no_grad
    def generate(self, labels_N):
        c_ND = self.class_embed(labels_N)
        ids_NL = self.decoder.generate(c_ND)
        z_NL = ids_NL.type(DTYPE) * (2 / (self.cfg.latent_bins - 1)) - 1
        x_ND = self.predictor_proj(z_NL)
        x_ND = self.predictor(x_ND + c_ND)
        p_ND = self.predictor_out(x_ND)
        return p_ND


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
        self.model = CondFSQ(cfg.model, self.metrics).to(self.device)
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

        # Generate and save samples
        labels_N = torch.arange(30, device=self.device) // 3
        imgs_ND = self.model.generate(labels_N)
        imgs = rearrange(imgs_ND, "N (H W) -> N H W", H=28, W=28)
        imgs = imgs * IMAGE_STD + IMAGE_MEAN
        imgs = imgs.clamp(0, 1) * 255
        imgs = imgs.type(torch.uint8)
        imgs = imgs.cpu().numpy()

        if self.step > 0:
            savepath = Path(__file__).parent / f"data/samples_step_{self.step + 1}.png"
        else:
            savepath = Path(__file__).parent / "data/samples_initial.png"

        imgs = rearrange(imgs, "(D M) H W -> (M H) (D W)", D=10)
        Image.fromarray(imgs).save(savepath)

        if self.step == 0:
            truth = {c: [] for c in range(10)}
            count = 0
            for img, label in mnist_loader(
                train=False, batch_size=1, epochs=1, device="cpu"
            ):
                if len(truth[int(label)]) < 3:
                    truth[int(label)].append(img.squeeze(0).numpy())
                    count += 1
                    if count == 30:
                        break
            truth = [truth[c][i] for c in range(10) for i in range(3)]
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
