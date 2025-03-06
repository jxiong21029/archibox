import argparse
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from sandbox.components import FusedLinear, ReLU2, RMSNorm
from sandbox.muon import Muon

log = logging.getLogger(__name__)


@dataclass
class Config:
    dim: int = 768
    depth: int = 12
    activation: str = "relu2"

    n_steps: int = 2000
    seq_len: int = 16 * 1024
    seq_len_valid: int = 32 * 1024
    valid_every: int = 125
    valid_tokens: int = 10 * 1024 * 1024
    vocab_size: int = 50257

    down_proj_muon: bool = True
    muon_lr: float = 0.05
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.05
    scalar_lr: float = 0.05
    embed_lr: float = 0.05
    lm_head_lr: float = 0.05
    value_lr: float = 0.05  # unused if down_proj_muon is True
    adamw_betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    adamw_weight_decay: float = 0.05
    cooldown_frac: float = 0.4


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        rotary_base: float,  # == maximum period of rotary emb / 2pi
        head_dim: int,
        activation: str = "relu2",
    ):
        super().__init__()
        self.dim = dim
        self.nheads = dim // head_dim
        self.head_dim = head_dim
        self.rotary_base = rotary_base

        self.attn_norm = RMSNorm(dim, affine=False)
        self.qkv_w = nn.Parameter(torch.randn(3, dim, dim).fmod(2) / math.sqrt(dim) / 2)
        self.qkv_w._iskey = True
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.o_proj = FusedLinear(dim, dim, scale=True)
        self.o_proj.weight._iskey = False
        self.o_proj.scale.data.zero_()

        freqs = (1 / rotary_base) ** torch.linspace(
            0, 1, head_dim // 4, dtype=torch.float32
        )
        self.freqs = nn.Buffer(
            torch.cat([freqs, freqs.new_zeros(head_dim // 4)]), persistent=False
        )

        if activation == "relu2":
            act = ReLU2()
        elif activation == "softmax":
            act = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"unknown activation: {activation}")

        self.mlp = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, 4 * dim),
            act,
            FusedLinear(4 * dim, dim, scale=True),
        )
        self.mlp[1]._iskey = True
        self.mlp[3]._iskey = False
        self.mlp[3].scale.data.zero_()

    def rotary(self, x_NTHD, shift: int):
        assert x_NTHD.ndim == 4
        assert shift >= 0
        T = x_NTHD.size(-3)

        theta_Td = torch.outer(
            torch.arange(shift, shift + T, device=self.freqs.device), self.freqs
        )
        cos_1T1d = torch.cos(theta_Td)[None, :, None, :]
        sin_1T1d = torch.sin(theta_Td)[None, :, None, :]
        x1_NTHd, x2_NTHd = x_NTHD.float().chunk(2, dim=-1)
        y1_NTHd = x1_NTHd * cos_1T1d + x2_NTHd * sin_1T1d
        y2_NTHd = x1_NTHd * (-sin_1T1d) + x2_NTHd * cos_1T1d
        return torch.cat([y1_NTHd, y2_NTHd], dim=-1).type_as(x_NTHD)

    def forward(self, x: Tensor, past_kv: tuple[Tensor, Tensor] | None):
        N, T, _ = x.shape
        residual = x
        qkv = self.attn_norm(x) @ self.qkv_w.flatten(0, 1).type_as(x).t()
        q, k, v = qkv.view(N, T, 3 * self.nheads, -1).chunk(3, dim=-2)
        q, k = self.qk_norm(q), self.qk_norm(k)
        shift = past_kv[0].size(1) if past_kv is not None else 0
        q, k = self.rotary(q, shift), self.rotary(k, shift)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x.transpose(1, 2).reshape(N, T, self.dim)
        x = residual + self.o_proj(x)

        x = x + self.mlp(x)
        return x, (k, v)


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        head_dim: int = 128,
        rotary_base: float = 1024.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dim=dim, rotary_base=rotary_base, head_dim=head_dim)
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor, kv_cache: list[tuple[Tensor, Tensor]] | None = None):
        assert x.ndim == 3
        if kv_cache is None:
            kv_cache = [None for _ in range(len(self.layers))]
        new_kv_cache = []
        for layer, past_kv in zip(self.layers, kv_cache, strict=True):
            x, new_kv = layer(x, past_kv)
            new_kv_cache.append(new_kv)
        return x, new_kv_cache


class GPT(nn.Module):
    def __init__(self, dim: int, depth: int, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed.weight.data.mul_(0.5)

        self.vocab_size = vocab_size
        self.decoder = Decoder(dim, depth)

        out_dim = ((vocab_size - 1) // 64) * 64 + 1
        self.lm_head = FusedLinear(dim, out_dim, scale=True)
        self.lm_head.scale.data.zero_()

    def forward(self, input_ids, target_ids):
        assert input_ids.ndim == 1
        x_NTD = self.embed(input_ids.unsqueeze(0))
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


class Metrics:
    def __init__(self, enabled: bool, use_wandb: bool = True, bufsize=128):
        self.enabled = enabled
        self.use_wandb = use_wandb

        self._n = defaultdict(int)
        self._mu = defaultdict(float)
        self._buf = []
        self._bufsize = bufsize

    @torch.no_grad
    def push(self, **metrics):
        if not self.enabled:
            return

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                assert v.numel() == 1, f"{k} shape={v.shape}"
                v = v.detach().to("cpu", non_blocking=True)
                self._buf.append((k, v))
            else:
                self._buf.append((k, v))

        if len(self._buf) > self._bufsize:
            self._sync()

    def _sync(self):
        """moves metric tensors on cuda device to cpu"""
        torch.cuda.synchronize()
        for k, v in self._buf:
            delta = float(v) - self._mu[k]
            self._n[k] += 1
            self._mu[k] += delta / self._n[k]
        self._buf.clear()

    def report(self):
        """averages metrics and logs to wandb"""
        if not self.enabled:
            return

        if len(self._buf) > 0:
            self._sync()

        results = {}
        for k in self._n:
            results[k] = self._mu[k]
        if self.use_wandb:
            wandb.log(results)
        else:
            log.info(str(results))

        self._n.clear()
        self._mu.clear()


def main():
    logging.basicConfig(level=logging.INFO)

    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    is_main_process = rank == 0
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    def log_once(s):
        if is_main_process:
            log.info(s)

    if is_main_process and not args.no_wandb:
        wandb.init(
            project="multi-task-agent",
            group="gpt",
            dir="runs/gpt",
            id=args.run_id,
            config=OmegaConf.to_container(OmegaConf.structured(cfg)),
            resume="must" if cfg.resume_from is not None else "never",
        )
    metrics = Metrics(enabled=is_main_process, use_wandb=not args.no_wandb)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = GPT(cfg.dim, cfg.depth, cfg.vocab_size)
    model.to(device)
    raw_model = model
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    muon_params = []
    value_params = []
    scalar_params = []
    for p in raw_model.decoder.parameters():
        if p.ndim < 2:
            scalar_params.append(p)
        elif cfg.down_proj_muon or p._iskey:
            muon_params.append(p)
        else:
            value_params.append(p)

    adamw_params = [
        dict(params=scalar_params, lr=cfg.scalar_lr),
        dict(params=[raw_model.embed], lr=cfg.embed_lr),
        dict(params=[raw_model.lm_head], lr=cfg.lm_head_lr),
    ]
    if value_params:
        adamw_params.append(dict(params=value_params, lr=cfg.value_lr))

    if is_main_process:
        n_muon_params = sum(p.numel() for p in muon_params)
        n_adamw_params = sum(p.numel() for group in adamw_params for p in group)
        n_adamw_tensors = sum(len(group["params"]) for group in adamw_params)
        n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total_tensors = len(list(model.parameters()))
        log.info(f"Muon: {n_muon_params:,} params across {len(muon_params)} tensors")
        log.info(f"AdamW: {n_adamw_params:,} params across {n_adamw_tensors} tensors")
        log.info(f"Total: {n_total_params:,} params across {n_total_tensors} tensors")
        if n_muon_params + n_adamw_params != n_total_params:
            log.warning(f"{n_muon_params=} + {n_adamw_params=} != {n_total_params=}")

    muon = ZeroRedundancyOptimizer(
        muon_params,
        Muon,
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        weight_decay=cfg.muon_weight_decay,
    )
    adamw = ZeroRedundancyOptimizer(
        adamw_params,
        torch.optim.AdamW,
        lr=cfg.adamw_lr,
        betas=cfg.adamw_betas,
        weight_decay=cfg.adamw_weight_decay,
    )
    optims = [muon, adamw]
    for opt in optims:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    def get_lr(step_):
        frac_done = step_ / cfg.n_steps
        return 0.1 + 0.9 * min(1, (1 - frac_done) / cfg.cooldown_frac)

    train_loader = distributed_data_generator(
        cfg.seq_len, rank, world_size, valid=False
    )
    for step in tqdm.trange(cfg.n_steps + 1, desc="training"):
        is_last_step = step == cfg.n_steps

        if is_last_step or (cfg.valid_every > 0 and step % cfg.valid_every == 0):
            with torch.no_grad():
                model.eval()
                for input_ids, target_ids in distributed_data_generator(
                    cfg.seq_len_valid, rank, world_size, valid=True
                ):
                    loss = model(input_ids, target_ids)
                    metrics.push(valid_loss=loss)
            metrics.report()
            model.train()

        if not is_last_step:
            lr_ratio = get_lr(step)
            for opt in optims:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lr_ratio

            input_ids, target_ids = next(train_loader)
            loss = model(input_ids, target_ids)
            metrics.push(train_loss=loss)

            loss.backward()
            for opt in optims:
                opt.step()
            for opt in optims:
                opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        dist.destroy_process_group()
