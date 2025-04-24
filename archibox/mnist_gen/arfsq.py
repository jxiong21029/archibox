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

from archibox.components import FusedLinear, ReLU2, RMSNorm
from archibox.metrics import Metrics
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.muon import Muon

metrics = Metrics(enabled=False, use_wandb=False)
log = logging.getLogger(__name__)


@dataclass
class MnistVectorFSQConfig:
    img_mean: float = 0.1307
    img_std: float = 0.3081

    dim: int = 1024
    mlp_dim: int = 2048
    latent_ndim: int = 12
    latent_bins: int = 8
    latent_group_size: int = 4
    prior_depth: int = 2
    encoder_depth: int = 2
    decoder_depth: int = 2

    mse_weight: float = 5.0
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float32
    )


@dataclass
class Config:
    use_wandb: bool = False
    savedir: str = str(Path(__file__).parent / "data/arfsq")

    model: MnistVectorFSQConfig = field(default_factory=MnistVectorFSQConfig)
    n_steps: int = 2000
    valid_every: int | None = 100
    batch_size: int = 4096

    muon_lr: float = 0.05
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


class FSQ(nn.Module):
    def __init__(self, in_dim: int, ndim: int, bins: int):
        super().__init__()
        self.bins = bins
        self.in_proj = nn.Sequential(
            RMSNorm(in_dim, affine=False), FusedLinear(in_dim, ndim)
        )

    def forward(self, x):
        latents = torch.tanh(self.in_proj(x).float())
        # [-1, 1] --> [0, bins - 1]
        latents_scaled = (latents + 1) * ((self.bins - 1) / 2)
        ids = torch.round(latents_scaled.detach())
        # Straight-through gradient estimator
        ids_ste = latents_scaled + (ids - latents_scaled).detach()
        # [0, bins - 1] --> [-1, 1]
        latents_ste = ids_ste * (2 / (self.bins - 1)) - 1
        return latents_ste.type_as(x), ids.long()


def make_embedding(num_embeddings: int, embedding_dim: int, dtype=None):
    embed = nn.Embedding(num_embeddings, embedding_dim)
    if dtype is not None:
        embed.to(dtype=dtype)
    embed.weight._is_embed = True
    embed.weight.data.mul_(0.5)
    return embed


class EmbedModulatedMLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.wk = FusedLinear(dim, mlp_dim)
        self.embed = make_embedding(vocab_size, mlp_dim)
        self.act = nn.SiLU()
        self.wv = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, c):
        scores = self.wk(self.norm(x)) * self.embed(c)
        scores = self.act(scores)
        return self.wv(scores)


def make_output_head(dim: int, out_dim: int):
    head = nn.Sequential(
        RMSNorm(dim, affine=False), FusedLinear(dim, out_dim, zero_init=True)
    )
    head[-1].weight._is_output = True
    return head


class VectorARPrior(nn.Module):
    def __init__(
        self, dim: int, mlp_dim: int, vocab_size: int, seq_len: int, prior_depth: int
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_vocab_size = ((self.vocab_size + 127) // 128) * 128

        self.prior = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(dim, affine=False),
                    FusedLinear(dim, mlp_dim),
                    ReLU2(),
                    FusedLinear(mlp_dim, dim, scale=True, zero_init=True),
                )
                for _ in range(prior_depth)
            ]
        )
        self.heads = nn.ModuleList(
            [make_output_head(dim, self.pad_vocab_size) for _ in range(seq_len)]
        )
        self.mlp_blocks = nn.ModuleList(
            [EmbedModulatedMLP(dim, mlp_dim, vocab_size) for _ in range(seq_len - 1)]
        )

    def loss(self, x_ND, token_ids_NL):
        for block in self.prior:
            x_ND = x_ND + block(x_ND)

        loss = 0
        for i in range(self.seq_len):
            logits = self.heads[i](x_ND)[..., : self.vocab_size]
            loss = loss + F.cross_entropy(
                logits.flatten(0, -2), token_ids_NL[..., i].flatten(), reduction="none"
            )
            if i < self.seq_len - 1:
                x_ND = x_ND + self.mlp_blocks[i](x_ND, token_ids_NL[..., i])
        return loss

    def sample(self, x_ND):
        for block in self.prior:
            x_ND = x_ND + block(x_ND)

        outputs = []
        for i in range(self.seq_len):
            logits = self.heads[i](x_ND)[..., : self.vocab_size]
            token_ids = Categorical(logits=logits).sample()
            outputs.append(token_ids)
            if i < self.seq_len - 1:
                x_ND = x_ND + self.mlp_blocks[i](x_ND, token_ids)
        return torch.stack(outputs, dim=-1)


class VectorModulatedMLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.wk = FusedLinear(dim, mlp_dim)
        self.wc = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.wv = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x, c):
        scores = self.wk(self.norm(x)) * self.wc(self.norm(c))
        scores = self.act(scores)
        return self.wv(scores)


class VectorModulatedMLPStack(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [VectorModulatedMLP(dim, mlp_dim) for _ in range(depth)]
        )

    def forward(self, x, c):
        for block in self.blocks:
            x = x + block(x, c)
        return x


class VectorARFSQ(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        data_dim: int,
        latent_ndim: int,
        latent_bins: int,
        latent_group_size: int,
        prior_depth: int,
        encoder_depth: int,
        decoder_depth: int,
    ):
        super().__init__()
        assert latent_ndim % latent_group_size == 0
        self.dim = dim
        self.data_dim = data_dim
        self.latent_ndim = latent_ndim
        self.latent_bins = latent_bins
        self.latent_group_size = latent_group_size
        self.vocab_size = latent_bins**latent_group_size

        self.encoder_in = FusedLinear(data_dim, dim, bias=True, gain=1.0)
        self.encoder = VectorModulatedMLPStack(dim, mlp_dim, encoder_depth)

        self.seq_len = latent_ndim // latent_group_size
        self.prior = VectorARPrior(
            dim, mlp_dim, self.vocab_size, self.seq_len, prior_depth=prior_depth
        )
        self.fsq_norm = RMSNorm(dim, affine=False)
        self.fsq = FSQ(dim, latent_ndim, latent_bins)
        self.decoder_in = FusedLinear(latent_ndim, dim, bias=True, gain=1.0)
        self.decoder = VectorModulatedMLPStack(dim, mlp_dim, decoder_depth)
        self.decoder_head = make_output_head(dim, data_dim)

        self._id_pows = nn.Buffer(
            latent_bins ** torch.arange(latent_group_size), persistent=False
        )

    def loss(self, data_ND, cond_ND):
        assert data_ND.size(-1) == self.data_dim
        assert cond_ND.size(-1) == self.dim

        data_emb = self.encoder(self.encoder_in(data_ND), cond_ND)
        latents_ND, latent_bins_ND = self.fsq(self.fsq_norm(data_emb))
        latent_bins_NLG = rearrange(
            latent_bins_ND,
            "... (L G) -> ... L G",
            L=self.seq_len,
            G=self.latent_group_size,
        )
        token_ids_N = (latent_bins_NLG * self._id_pows).sum(dim=-1)
        xent_N = self.prior.loss(cond_ND, token_ids_N)

        x_ND = self.decoder_in(latents_ND)
        x_ND = self.decoder(x_ND, cond_ND)
        reconstruction_ND = self.decoder_head(x_ND)
        mse_ND = (data_ND - reconstruction_ND).pow(2)
        return xent_N, mse_ND

    def generate(self, cond_ND):
        token_ids_NL = self.prior.sample(cond_ND)
        assert token_ids_NL.shape == cond_ND.shape[:-1] + (self.seq_len,)
        latents_NLG = (token_ids_NL.unsqueeze(-1) // self._id_pows) % self.latent_bins
        latents_ND = rearrange(
            latents_NLG,
            "... L G -> ... (L G)",
            L=self.seq_len,
            G=self.latent_group_size,
        )
        latents_ND = latents_ND.type_as(cond_ND) * (2 / (self.latent_bins - 1)) - 1
        x_ND = self.decoder_in(latents_ND)
        x_ND = self.decoder(x_ND, cond_ND)
        reconstruction_ND = self.decoder_head(x_ND)
        return reconstruction_ND


class MnistVectorFSQ(nn.Module):
    def __init__(self, cfg: MnistVectorFSQConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = self.cfg.dtype

        self.digit_embed = make_embedding(10, cfg.dim)
        self.vector_fsq = VectorARFSQ(
            dim=cfg.dim,
            mlp_dim=cfg.mlp_dim,
            data_dim=28 * 28,
            latent_ndim=cfg.latent_ndim,
            latent_bins=cfg.latent_bins,
            latent_group_size=cfg.latent_group_size,
            prior_depth=cfg.prior_depth,
            encoder_depth=cfg.encoder_depth,
            decoder_depth=cfg.decoder_depth,
        )

    def forward(self, images, labels):
        images = images.type(self.dtype) / 255.0
        images = images.flatten(1, -1)
        images = (images - self.cfg.img_mean) / self.cfg.img_std

        cond = self.digit_embed(labels).type(self.dtype)
        xent, mse = self.vector_fsq.loss(images, cond)
        xent = xent.mean()
        mse = mse.mean()
        loss = xent + self.cfg.mse_weight * mse
        metrics.push(xent=xent, mse=mse, loss=loss)
        return loss

    @torch.no_grad
    def generate(self, labels_N):
        cond = self.digit_embed(labels_N).type(self.dtype)
        images = self.vector_fsq.generate(cond)
        return images


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
        self.model = MnistVectorFSQ(cfg.model).to(self.device)

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
        metrics.push(relative_lr=relative_lr)

    def train_step(self):
        self.schedule_lr()

        images, labels = next(self.train_loader)
        # TODO
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
        metrics.context = "valid_"
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
        metrics.push(pixel_counts_distance=distance)

        results = {}
        for k in metrics.n:
            if metrics.n[k] == 0:
                continue
            if isinstance(metrics.mean[k], torch.Tensor):
                results[k] = metrics.mean[k].to("cpu", non_blocking=True)
            else:
                results[k] = metrics.mean[k]
        print("\n" + str(results))

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
