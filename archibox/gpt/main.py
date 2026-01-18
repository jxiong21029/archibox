import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

import einops as eo
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from bnnp import DistMuon, Metrics, parse_config
from bnnp.nn import Embedding, FusedLinear, Output, RMSNorm, RoPE1d, mpparam
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)
rank0logger = logging.getLogger(f"{__name__}.rank0")
metrics = Metrics(use_wandb=False, enabled=False, use_cuda_events=True)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_name: str | None = None
    runs_dir: str = str(Path(__file__).parent / "runs")
    n_steps: int = 2000
    seq_len: int = 64 * 1024
    valid_every: int = 125
    valid_batches: int = 160

    vocab_size: int = 50257
    dim: int = 768
    head_dim: int = 64
    mlp_dim: int = 3072
    depth: int = 12
    temperature: Literal["affine", "scalar"] | None = "scalar"
    exnorm: bool = False

    muon_lr: float = 0.01
    muon_mu: float = 0.95
    embed_lr: float = 0.003
    scalar_lr: float = 0.003
    low_rank_lr: float = 0.003
    adamw_betas: tuple[float, float] = (0.95, 0.99)
    weight_decay: float = 0.01
    lr_cooldown_start: int = 1200
    lr_cooldown_ratio: float = 0.0

    compile_mode: str | None = "default"
    dtype: str = "bfloat16"
    use_wandb: bool = True
    seed: int = 0


class DecoderAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope: RoPE1d,
        temperature: Literal["affine", "scalar"] | None,
        exnorm: bool,
    ):
        assert head_dim % 4 == 0
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.temperature = temperature
        self.exnorm = exnorm

        self.in_norm = RMSNorm(dim, affine=False)
        self.qkv_weight = mpparam(3, dim, dim)
        if temperature == "scalar":
            self.temp_invm1 = nn.Parameter(torch.zeros(head_dim))
            self.temp_invm1._group = "scalar"  # ty: ignore
        self.q_norm = RMSNorm(head_dim, affine=temperature == "affine")
        self.k_norm = RMSNorm(head_dim, affine=temperature == "affine")
        self.rope = rope
        self.o_proj = FusedLinear(dim, dim, zero_init=True)

    def forward(
        self, input_BTD: Tensor, rotations: tuple[Tensor, Tensor], block_mask: BlockMask
    ):
        B, T, _ = input_BTD.shape
        x_BTD = self.in_norm(input_BTD)
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_BTD)
        q_BThd, k_BThd, v_BThd = (
            (x_BTD @ qkv_weight.t())
            .view(B, T, 3 * self.n_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        if self.temperature == "scalar":
            scale = self.temp_invm1.add(1).sqrt().type_as(input_BTD)
            q_BThd = self.rope(self.q_norm(q_BThd) * scale, *rotations)
            k_BThd = self.rope(self.k_norm(k_BThd) * scale, *rotations)
        else:
            q_BThd = self.rope(self.q_norm(q_BThd), *rotations)
            k_BThd = self.rope(self.k_norm(k_BThd), *rotations)

        o_BhTd = flex_attention(
            q_BThd.transpose(1, 2),
            k_BThd.transpose(1, 2),
            v_BThd.transpose(1, 2),
            block_mask=block_mask,
        )
        o_BTD = eo.rearrange(o_BhTd, "B h T d -> B T (h d)")
        o_BTD = self.o_proj(o_BTD)

        if self.exnorm:
            post_scale = o_BTD.square().mean(dim=-1, keepdim=True).add(1).rsqrt()
            return post_scale * (input_BTD + o_BTD)
        return input_BTD + o_BTD


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, exnorm: bool):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.exnorm = exnorm

        self.in_norm = RMSNorm(dim, affine=False)
        self.up_proj = FusedLinear(dim, mlp_dim)
        self.down_proj = FusedLinear(mlp_dim, dim, zero_init=True)

    def forward(self, input_BTD: Tensor):
        x_BTD = self.in_norm(input_BTD)
        x_BTD = self.up_proj(x_BTD)
        x_BTD = F.relu(x_BTD).square()
        x_BTD = self.down_proj(x_BTD)
        if self.exnorm:
            post_scale = x_BTD.square().mean(dim=-1, keepdim=True).add(1).rsqrt()
            return post_scale * (input_BTD + x_BTD)
        return input_BTD + x_BTD


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        head_dim: int,
        mlp_dim: int,
        depth: int,
        temperature: Literal["affine", "scalar"] | None,
        exnorm: bool,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, dim)

        self.rope = RoPE1d(
            head_dim=head_dim,
            min_freq=0.001,
            max_freq=1.0,
            p_zero_freqs=0.5,
        )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.ModuleDict()
            block["attn"] = DecoderAttention(
                dim=dim,
                head_dim=head_dim,
                rope=self.rope,
                temperature=temperature,
                exnorm=exnorm,
            )
            block["mlp"] = MLP(dim=dim, mlp_dim=mlp_dim, exnorm=exnorm)
            self.blocks.append(block)
        self.out_head = Output(dim, vocab_size, pad_to=64)

    def precompute_rotations(
        self,
        seq_len: int,
        device: torch.device | str,
        dtype=torch.dtype,
    ):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        cos, sin = self.rope.precompute_rotations(pos)
        rotations = (cos.to(dtype), sin.to(dtype))
        self.dtype = dtype

        return rotations

    def forward(self, input_ids_BT: Tensor, rotations: tuple[Tensor, Tensor]):
        B, T = input_ids_BT.size()
        assert B == 1

        x_BTD = self.embed(input_ids_BT).to(self.dtype)

        docs = (input_ids_BT[0] == 50256).cumsum(0)

        def mask_fn(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx < 1024
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & window_mask & document_mask

        block_mask = create_block_mask(
            mask_fn, B=None, H=None, Q_LEN=T, KV_LEN=T, device=input_ids_BT.device
        )

        for block in self.blocks:
            assert isinstance(block, nn.ModuleDict)
            x_BTD = block["attn"](x_BTD, rotations, block_mask=block_mask)
            x_BTD = block["mlp"](x_BTD)
        assert x_BTD.dtype == self.dtype

        logits = self.out_head(x_BTD)
        metrics = {}
        with torch.no_grad():
            metrics["residual_rms"] = x_BTD.square().mean(dim=-1).sqrt().mean()
        return logits, metrics


def _load_data_shard(file: Path):
    assert file.is_file()
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


def distributed_data_generator(
    seq_len: int, rank: int, world_size: int, valid: bool, device: torch.device
):
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
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        ptr += seq_len * world_size
        yield inputs, targets


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(name)s:%(levelname)s] %(message)s",
        )

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main_process = self.rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.world_size > 1
        self.device = torch.device("cuda", self.local_rank)
        metrics.enabled = self.is_main_process
        torch.cuda.set_device(self.device)
        local_seed = 0x30005 * cfg.seed + self.rank
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed(local_seed)
        rank0logger.setLevel(logging.INFO if self.is_main_process else logging.ERROR)
        rank0logger.info("Config: " + str(json.dumps(cfg.model_dump(), indent=4)))

        run_base_dir = Path(cfg.runs_dir)
        run_base_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main_process and cfg.use_wandb:
            wandb.init(
                entity="archibox",
                project="nanogpt",
                name=cfg.run_name,
                dir=run_base_dir,
                config=cfg.model_dump(),
                resume="never",
            )
            metrics.use_wandb = True
            assert wandb.run is not None
            self.run_dir = Path(wandb.run.dir)
        else:
            self.run_dir = run_base_dir / "local"
        if self.is_main_process:
            self.run_dir.mkdir(exist_ok=True)
        if self.is_distributed:
            dist.init_process_group(backend="nccl", device_id=self.device)
        else:
            assert self.rank == self.local_rank == 0

        if self.is_main_process:
            self.info = dict(
                commit_hash=subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], text=True
                ).strip(),
                diff=subprocess.check_output(
                    ["git", "diff", "HEAD"], text=True
                ).strip(),
                hostname=subprocess.check_output(["hostname"], text=True).strip(),
                nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True).strip(),
            )
            rank0logger.info(f"git commit hash: {self.info['commit_hash']}")
            rank0logger.info(f"git diff:\n{self.info['diff']}")
            rank0logger.info(f"running on host: {self.info['hostname']}")
            rank0logger.info(f"nvidia-smi:\n{self.info['nvidia_smi']}")

        self.dtype = dict(
            bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32
        )[cfg.dtype]

        self.train_loader = distributed_data_generator(
            cfg.seq_len, self.rank, self.world_size, valid=False, device=self.device
        )

        model = GPT(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            head_dim=cfg.head_dim,
            mlp_dim=cfg.mlp_dim,
            depth=cfg.depth,
            temperature=cfg.temperature,
            exnorm=cfg.exnorm,
        ).to(self.device)
        self.rotations = model.precompute_rotations(
            seq_len=cfg.seq_len, device=self.device, dtype=self.dtype
        )
        self.raw_model = model
        if cfg.compile_mode is not None:
            model.compile(mode=cfg.compile_mode, dynamic=False)
        if self.is_distributed:
            model = DDP(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        self.model = model

        param_groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            name_shape = f"{name!r} {tuple(param.shape)}"

            old_attrs = ("_is_scalar", "_is_embed", "_is_output", "_no_weight_decay")
            if any(hasattr(param, attr) for attr in old_attrs):
                rank0logger.warning(
                    f"Parameter {name_shape} has deprecated optimizer attributes: "
                    + f"{[attr for attr in old_attrs if hasattr(param, attr)]}. "
                    + "Use _group instead."
                )

            if not hasattr(param, "_group") or param._group == "muon":  # pyright: ignore
                if param.ndim >= 2:
                    group = "muon"
                elif not hasattr(param, "_group"):
                    raise ValueError(
                        f"Parameter {name_shape} with <2 dims must be assigned a _group."
                    )
                else:
                    raise ValueError(
                        f"Parameter {name_shape} with <2 dims cannot be assigned group 'muon'."
                    )
            elif param._group not in ("embed", "scalar", "low_rank"):  # pyright: ignore
                raise ValueError(
                    f"Parameter {name_shape} has invalid _group attribute: {param._group}"  # pyright: ignore
                )
            else:
                group = param._group

            rank0logger.info(f"Parameter {name_shape} assigned to group {group!r}.")
            param_groups.setdefault(group, []).append(param)

        total_params, trainable_params, total_tensors = 0, 0, 0
        for group, params in param_groups.items():
            n_params = sum(p.numel() for p in params)
            n_trainable = sum(p.numel() for p in params if p.requires_grad)
            total_params += n_params
            trainable_params += n_trainable
            total_tensors += len(params)
            rank0logger.info(
                f"> Group {group}: {n_params:,} parameters ({n_trainable:,} trainable) over {len(params):,} tensors"
            )
        rank0logger.info(
            f"Total: {total_params:,} parameters ({trainable_params:,} trainable) over {total_tensors:,} tensors"
        )

        parameters = (
            {
                "params": params,
                "algorithm": "muon" if group == "muon" else "adamw",
                "lr": {
                    "muon": cfg.muon_lr,
                    "embed": cfg.embed_lr,
                    "scalar": cfg.scalar_lr,
                    "low_rank": cfg.low_rank_lr,
                }[group],
            }
            for group, params in param_groups.items()
        )
        self.optimizer = DistMuon(
            parameters,
            lr=self.cfg.muon_lr,
            mu=self.cfg.muon_mu,
            adamw_betas=cfg.adamw_betas,
            weight_decay=self.cfg.weight_decay,
            nesterov=True,
            lr_scaling="rms",
        )
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        self.step = 0
        metrics.context = "train_"

    def run(self):
        with tqdm.tqdm(
            total=self.cfg.n_steps, desc="train", ncols=88, mininterval=3.0
        ) as pbar:
            if self.step > 0:
                pbar.update(self.step)
            while self.step < self.cfg.n_steps:
                cd_start = self.cfg.lr_cooldown_start
                if cd_start is not None and self.step > cd_start:
                    cd_frac = (self.step - cd_start) / (self.cfg.n_steps - cd_start)
                    relative_lr = 1.0 - cd_frac * (1 - self.cfg.lr_cooldown_ratio)
                    for group in self.optimizer.param_groups:
                        group["lr"] = group["initial_lr"] * relative_lr
                else:
                    relative_lr = 1.0
                metrics.push(relative_lr=relative_lr)

                self.train_step()
                if (self.step + 1) % self.cfg.valid_every == 0:
                    self.valid_epoch()
                metrics.commit(_step=self.step + 1)

                self.step += 1
                pbar.update(1)

    def train_step(self):
        metrics.tick("load_batch")
        inputs_ids, target_ids = next(self.train_loader)

        metrics.tick("forward")
        logits, fwd_metrics = self.model(inputs_ids.unsqueeze(0), self.rotations)
        loss = F.cross_entropy(logits[0].float(), target_ids)
        assert torch.isfinite(loss)
        metrics.push(loss=loss, **fwd_metrics)

        metrics.tick("backward")
        loss.backward()

        metrics.tick("optim")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()
        metrics.context = "valid_"

        valid_loader = distributed_data_generator(
            self.cfg.seq_len, self.rank, self.world_size, valid=True, device=self.device
        )
        for _ in tqdm.trange(
            self.cfg.valid_batches, desc="valid", ncols=88, mininterval=3.0
        ):
            input_ids, target_ids = next(valid_loader)
            logits, fwd_metrics = self.model(input_ids.unsqueeze(0), self.rotations)
            loss = F.cross_entropy(logits[0].float(), target_ids)
            metrics.push(loss=loss, **fwd_metrics)

        self.model.train()
        metrics.context = "train_"


@parse_config
def main(cfg: Config):
    try:
        trainer = Trainer(cfg)
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
