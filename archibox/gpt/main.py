import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP

from archibox.components import FusedLinear, ReLU2, RMSNorm
from archibox.metrics import Metrics
from archibox.muon import Muon

log = logging.getLogger(__name__)
flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")


@dataclass
class Config:
    n_steps: int = 2000
    seq_len: int = 4 * 1024  # gpu poor moment
    seq_len_valid: int = 4 * 1024
    valid_every: int = 125
    valid_tokens: int = 10 * 1024 * 1024

    dim: int = 768
    depth: int = 12
    vocab_size: int = 50257

    muon_lr: float = 0.05
    muon_mu: float = 0.95
    muon_wd: float = 0.05
    scalar_lr: float = 0.003
    embeds_lr: float = 0.003
    output_lr: float = 0.003
    adamw_mu1: float = 0.95
    adamw_mu2: float = 0.99
    adamw_wd: float = 0.05
    lr_cooldown_start: int = 1200
    lr_cooldown_ratio: float = 0.1


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        window_size: int,
        rotary_base: float,
        head_dim: int,
        mlp_mult: float,
    ):
        assert head_dim % 4 == 0
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len  # training sequence length
        self.window_size = window_size  # max number of tokens to attend to
        self.rotary_base = rotary_base  # maximum period of rotary / 2pi
        self.nheads = dim // head_dim
        self.head_dim = head_dim

        self.attn_norm = RMSNorm(dim, affine=False)
        self.qkv_w = nn.Parameter(torch.randn(3, dim, dim).fmod(2) / math.sqrt(dim) / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.o_proj = FusedLinear(dim, dim, scale=True)
        self.o_proj.scale.data.zero_()

        freqs = (1 / rotary_base) ** torch.linspace(
            0, 1, head_dim // 4, dtype=torch.float32
        )
        # Used during inference
        self.freqs_d = nn.Buffer(
            torch.cat((freqs, freqs.new_zeros(head_dim // 4))), persistent=False
        )
        theta_1T1d = torch.outer(torch.arange(seq_len), self.freqs_d).reshape(
            1, seq_len, 1, head_dim // 2
        )
        # Used during training
        self.cos_1T1d = nn.Buffer(torch.cos(theta_1T1d), persistent=False)
        self.sin_1T1d = nn.Buffer(torch.sin(theta_1T1d), persistent=False)

        def sliding_window_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx < kv_idx + window_size)

        self.block_mask = create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
        )

        self.mlp = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, round(mlp_mult * dim)),
            ReLU2(),
            FusedLinear(round(mlp_mult * dim), dim, scale=True),
        )
        self.mlp[-1].scale.data.zero_()

    def rotary(self, x_NTHD):
        """Training rotary embedding (known seq_len, cached cos/sin)"""
        assert x_NTHD.ndim == 4
        assert x_NTHD.size(1) == self.seq_len
        x1_NTHd, x2_NTHd = x_NTHD.float().chunk(2, dim=-1)
        y1_NTHd = x1_NTHd * self.cos_1T1d + x2_NTHd * self.sin_1T1d
        y2_NTHd = x1_NTHd * (-self.sin_1T1d) + x2_NTHd * self.cos_1T1d
        return torch.cat([y1_NTHd, y2_NTHd], dim=-1).type_as(x_NTHD)

    def inference_rotary(self, x_NHD, t: int):
        """Inference rotary embedding (one input token)"""
        assert x_NHD.ndim == 3
        assert t >= 0
        theta_d = t * self.freqs_d
        cos_d = torch.cos(theta_d)
        sin_d = torch.sin(theta_d)
        x1_NHd, x2_NHd = x_NHD.float().chunk(2, dim=-1)
        y1_NHd = x1_NHd * cos_d + x2_NHd * sin_d
        y2_NHd = x1_NHd * (-sin_d) + x2_NHd * cos_d
        return torch.cat([y1_NHd, y2_NHd], dim=-1).type_as(x_NHD)

    def forward(self, x_NTD: Tensor):
        """Training forward (fixed seq_len, no past_kv)"""
        N, T, _ = x_NTD.shape
        residual = x_NTD
        qkv = self.attn_norm(x_NTD) @ self.qkv_w.flatten(0, 1).type_as(x_NTD).t()
        q_NTHD, k_NTHD, v_NTHD = qkv.view(N, T, 3 * self.nheads, -1).chunk(3, dim=2)
        q_NTHD, k_NTHD = self.qk_norm(q_NTHD), self.qk_norm(k_NTHD)
        q_NTHD, k_NTHD = self.rotary(q_NTHD), self.rotary(k_NTHD)
        x_NHTD = flex_attention(
            q_NTHD.transpose(1, 2),
            k_NTHD.transpose(1, 2),
            v_NTHD.transpose(1, 2),
            block_mask=self.block_mask,
        )
        x_NTD = x_NHTD.transpose(1, 2).reshape(N, T, self.dim)
        x_NTD = residual + self.o_proj(x_NTD)

        x_NTD = x_NTD + self.mlp(x_NTD)
        return x_NTD

    def predict(self, x_ND: Tensor, past_kv: tuple[Tensor, Tensor] | None, t: int):
        """Inference (one input token, optional past_kv). Returns new kv."""
        N, _ = x_ND.shape
        residual = x_ND
        qkv = self.attn_norm(x_ND) @ self.qkv_w.flatten(0, 1).type_as(x_ND).t()
        q_NHD, k_NHD, v_NHD = qkv.view(N, 3 * self.nheads, -1).chunk(3, dim=1)
        q_NHD, k_NHD = self.qk_norm(q_NHD), self.qk_norm(k_NHD)
        q_NHD = self.inference_rotary(q_NHD, t)
        k_NHD = self.inference_rotary(k_NHD, t)
        if past_kv is not None:
            past_k_NHTD, past_v_NHTD = past_kv
            k_NHTD = torch.cat([past_k_NHTD, k_NHD.unsqueeze(2)], dim=2)
            v_NHTD = torch.cat([past_v_NHTD, v_NHD.unsqueeze(2)], dim=2)
        else:
            k_NHTD = k_NHD.unsqueeze(2)
            v_NHTD = v_NHD.unsqueeze(2)
        x_NHTD = F.scaled_dot_product_attention(q_NHD.unsqueeze(2), k_NHTD, v_NHTD)
        x_ND = x_NHTD[:, :, -1, :].flatten(1, 2)
        x_ND = residual + self.o_proj(x_ND)

        x_ND = x_ND + self.mlp(x_ND)
        return x_ND, (
            k_NHTD[:, :, -self.window_size + 1 :],
            v_NHTD[:, :, -self.window_size + 1 :],
        )


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        seq_len: int,
        window_size: int,
        head_dim: int = 64,
        mlp_mult: float = 4.0,
        rotary_base: float = 1024.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=dim,
                    seq_len=seq_len,
                    window_size=window_size,
                    rotary_base=rotary_base,
                    head_dim=head_dim,
                    mlp_mult=mlp_mult,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x_NTD: Tensor):
        """Training forward (fixed seq_len, no kv_cache)"""
        for layer in self.layers:
            x_NTD = layer(x_NTD)
        return x_NTD

    def predict(self, x: Tensor, kv_cache: list[tuple[Tensor, Tensor]] | None, t: int):
        """Inference (one input token, optional kv_cache). Returns new kv_cache"""
        if kv_cache is None:
            kv_cache = [None for _ in range(len(self.layers))]
        new_kv_cache = []

        for layer, past_kv in zip(self.layers, kv_cache, strict=True):
            x, new_kv = layer.predict(x, past_kv, t)
            new_kv_cache.append(new_kv)
        return x, new_kv_cache


class GPT(nn.Module):
    def __init__(self, dim: int, depth: int, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed.weight.data.mul_(0.5)

        self.vocab_size = vocab_size
        self.decoder = Decoder(dim, depth)

        out_dim = (vocab_size + 63) // 64 * 64
        self.lm_head = FusedLinear(dim, out_dim, scale=True)
        self.lm_head.scale.data.zero_()

    def forward(self, input_ids, target_ids):
        assert input_ids.ndim == 1
        x_NTD = self.embed(input_ids.unsqueeze(0)).to(torch.bfloat16)
        x_NTD, _ = self.decoder(x_NTD)
        logits_NTD = self.lm_head(x_NTD)
        loss = F.cross_entropy(logits_NTD.squeeze(0), target_ids)
        return loss


def _load_data_shard(file: Path):
    # header is 256 int32
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        # avoid pin_memory copy by @YouJiacheng
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(seq_len: int, rank: int, world_size: int, valid: bool):
    basedir = Path(__file__).parent / "data/fineweb10B"
    files = sorted(basedir.glob(f"fineweb_{'val' if valid else 'train'}_*.bin"))
    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    ptr = 0
    while True:
        if ptr + seq_len * world_size + 1 >= len(tokens):
            tokens = _load_data_shard(next(file_iter))
            ptr = 0
        buf = tokens[ptr + rank * seq_len : ptr + (rank + 1) * seq_len + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        ptr += seq_len * rank
        yield inputs, targets


class Trainer:
    def __init__(self, cfg: Config, use_wandb: bool, do_compile: bool):
        self.cfg = cfg
        self.use_wandb = use_wandb

        self.rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        self.is_main_process = self.rank == 0
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl", device_id=self.device)

        if self.is_main_process and use_wandb:
            Path("runs/gpt").mkdir(exist_ok=True)
            wandb.init(
                project="archibox",
                group="gpt",
                dir="runs/gpt",
                config=OmegaConf.to_container(OmegaConf.structured(self.cfg)),
            )

        torch.manual_seed(self.rank)
        torch.cuda.manual_seed(self.rank)

        self.metrics = Metrics(enabled=self.is_main_process, use_wandb=use_wandb)
        self.metrics.context = "train_"

        self.train_loader = distributed_data_generator(
            cfg.seq_len, self.rank, self.world_size, valid=False
        )

        model = GPT(cfg.dim, cfg.depth, cfg.vocab_size).to(self.device)
        self.raw_model = model
        if do_compile:
            model = torch.compile(model)
        self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        scalar_params = []
        embeds_params = []
        output_params = []
        muon_params = []
        for name, p in self.raw_agent.named_parameters():
            shape = tuple(p.shape)
            if not p.requires_grad:
                self.log_once(f"{name} {shape} requires_grad=False, skipped")
                continue
            elif p.ndim < 2:
                self.log_once(f"{name} {shape} assigned to AdamW")
                scalar_params.append(p)
            elif hasattr(p, "_is_embed") and p._is_embed:
                self.log_once(f"{name} {shape} (_is_embed=True) assigned to AdamW")
                embeds_params.append(p)
            elif hasattr(p, "_is_output") and p._is_output:
                self.log_once(f"{name} {shape} (_is_output=True) assigned to AdamW")
                output_params.append(p)
            else:
                # _ortho is deprecated but checking here just in case
                assert not hasattr(p, "_ortho")
                self.log_once(f"{name}{shape} assigned to Muon")
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

        if self.is_main_process and self.use_wandb:
            OmegaConf.save(self.cfg, Path(wandb.run.dir) / "cfg.yaml")

        self.epoch = 0
        self.step = 0

        self.log_once(f"starting @ step={self.step}, epoch={self.epoch}")

    def log_once(self, s):
        if self.is_main_process:
            log.info(s)

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
        inputs_ids, target_ids = next(self.train_loader)
        loss = self.model(inputs_ids, target_ids)
        self.metrics.push(train_loss=loss)
        loss.backward()
        for optim in self.optims:
            optim.step()
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

    @torch.no_grad
    def valid_epoch(self):
        self.agent.eval()
        for input_ids, target_ids in distributed_data_generator(
            self.cfg.seq_len_valid, self.rank, self.world_size, valid=True
        ):
            self.metrics.push(valid_loss=self.model(input_ids, target_ids))
        self.metrics.report()
        self.agent.train()

    def run(self):
        with tqdm.tqdm(
            total=self.cfg.n_steps, desc="training", mininterval=1.0, smoothing=0.05
        ) as progress_bar:
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = Config()

    try:
        trainer = Trainer(cfg, not args.no_wandb, args.compile)
        trainer.run()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
