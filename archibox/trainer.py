import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from bnnp import Metrics, Muon, auto_split_muon_params
from pydantic import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

log = logging.getLogger(__name__)


class TrainerConfig(BaseModel):
    run_dir: str | None
    resume_from_ckpt: str | None = None
    debug: bool = False

    n_steps: int
    valid_every: int | None
    save_every: int | None
    ckpt_best_key: str | None = None
    micro_batch_size: int
    train_loader_workers: int
    valid_loader_workers: int

    muon_lr: float
    muon_wd: float
    muon_mu: float = 0.9
    scalar_lr: float
    embeds_lr: float
    output_lr: float
    adamw_wd: float
    adamw_mu1: float = 0.9
    adamw_mu2: float = 0.99
    lr_cooldown_start: int | None = None
    lr_cooldown_ratio: float = 0.1
    grad_accum_steps: int = 1


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        # model forward should expect one argument, the batch, and return a scalar loss
        model: nn.Module,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        metrics: Metrics | None = None,
    ):
        assert cfg.lr_cooldown_start is None or cfg.lr_cooldown_start < cfg.n_steps
        self.cfg = cfg
        self.raw_model = model

        self.metrics = metrics if metrics is not None else Metrics(enabled=False)

        self.rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main_process = self.rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            dist.init_process_group(backend="nccl", device_id=self.device)
            if torch.cuda.device_count() != int(os.environ["LOCAL_WORLD_SIZE"]):
                log.warning(
                    f"device_count={torch.cuda.device_count()} != LOCAL_WORLD_SIZE={os.environ['LOCAL_WORLD_SIZE']}"
                )

        if metrics is not None:
            metrics.enabled = self.is_main_process

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.log(
            f"dataset lengths: train {len(train_dataset):,} || valid {len(valid_dataset)}"
        )
        self.train_sampler = (
            DistributedSampler(train_dataset, shuffle=True)
            if self.world_size > 1
            else None
        )
        self.train_loader = StatefulDataLoader(
            train_dataset,
            batch_size=cfg.micro_batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            drop_last=True,
            num_workers=cfg.train_loader_workers,
            pin_memory=cfg.train_loader_workers > 0,
            timeout=600 if cfg.train_loader_workers > 0 else 0,
        )
        self.valid_sampler = (
            DistributedSampler(valid_dataset, shuffle=False)
            if self.world_size > 1
            else None
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.micro_batch_size,
            sampler=self.valid_sampler,
            drop_last=True,
            num_workers=cfg.valid_loader_workers,
            pin_memory=cfg.valid_loader_workers > 0,
            timeout=600 if cfg.valid_loader_workers > 0 else 0,
        )

        model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            self.model = model

        muon_p, scalar_p, embeds_p, output_p = auto_split_muon_params(self.raw_model)
        adamw_params = [
            dict(params=scalar_p, lr=cfg.scalar_lr),
            dict(params=embeds_p, lr=cfg.embeds_lr),
            dict(params=output_p, lr=cfg.output_lr),
        ]
        self.muon = Muon(
            muon_p, lr=cfg.muon_lr, momentum=cfg.muon_mu, weight_decay=cfg.muon_wd
        )
        self.adamw = torch.optim.AdamW(
            adamw_params,
            betas=[cfg.adamw_mu1, cfg.adamw_mu2],
            weight_decay=cfg.adamw_wd,
        )
        self.optims = [self.muon, self.adamw]
        for opt in self.optims:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        if cfg.resume_from_ckpt is not None:
            assert Path(cfg.resume_from_ckpt).exists()
            self.load_checkpoint(cfg.resume_from_ckpt)
        else:
            self.epoch = 0
            self.step = 0
            self.ckpt_best_value = float("inf")

            if self.is_main_process and self.cfg.run_dir is not None:
                with open(Path(cfg.run_dir) / "cfg.json", "w") as f:
                    f.write(cfg.model_dump_json(indent=4))
        self.ckpt_last_value = None
        # If loading from checkpoint, fork CPU RNG so that it doesn't get changed here.
        with torch.random.fork_rng(
            devices=[],  # Don't fork any GPU RNGs
            enabled=cfg.resume_from_ckpt is not None,
        ):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.epoch)
            self.train_loader_iter = iter(self.train_loader)

        self.log(f"starting @ step={self.step}, epoch={self.epoch}")

    def log(self, s, level=logging.INFO):
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
        self.metrics.push(relative_lr=relative_lr)

    def on_epoch_end(self):
        self.log(f"starting epoch {self.epoch}")

    def train_step(self):
        self.schedule_lr()

        self.metrics.tick("load_batch")
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            self.epoch += 1
            self.on_epoch_end()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.epoch)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        batch = tree_map(
            lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch
        )

        self.metrics.tick("forward")
        loss = self.model(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        if self.cfg.grad_accum_steps > 1:
            loss = loss / self.cfg.grad_accum_steps

        self.metrics.tick("backward")
        loss.backward()

        if (self.step + 1) % self.cfg.grad_accum_steps == 0:
            self.metrics.tick("optim")
            for optim in self.optims:
                optim.step()
            for optim in self.optims:
                optim.zero_grad(set_to_none=True)

        self.metrics.tick(None)

    def on_validation_end(self):
        pass

    def valid_epoch(self):
        self.model.eval()
        self.metrics.context = "valid_"
        torch.set_grad_enabled(False)

        self.metrics.tick("load_batch")
        for i, batch in enumerate(
            tqdm.tqdm(self.valid_loader, desc="validation", mininterval=5.0)
        ):
            batch = tree_map(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch
            )
            self.metrics.tick("forward")
            self.model(batch)
            if i < len(self.valid_loader) - 1:
                self.metrics.tick("load_batch")
        self.metrics.tick(None)

        if self.is_main_process and self.cfg.ckpt_best_key is not None:
            if self.cfg.ckpt_best_key not in self.metrics.mean:
                self.log(
                    f"ckpt_best_key={self.cfg.ckpt_best_key} not found in metrics",
                    level=logging.WARNING,
                )
            self.ckpt_last_value = self.metrics.mean.get(self.cfg.ckpt_best_key, None)

        self.on_validation_end()

        self.metrics.report()
        self.model.train()
        self.metrics.context = "train_"
        torch.set_grad_enabled(True)

    def run(self):
        with tqdm.tqdm(
            total=self.cfg.n_steps, desc="training", mininterval=5.0
        ) as progress_bar:
            if self.step == 0:
                self.log("running initial validation epoch")
                self.valid_epoch()
            else:
                progress_bar.update(self.step)
            while self.step < self.cfg.n_steps:
                self.train_step()

                new_ckpts = []
                if (
                    self.cfg.valid_every is not None
                    and (self.step + 1) % self.cfg.valid_every == 0
                ) or self.step + 1 == self.cfg.n_steps:
                    self.valid_epoch()

                    if self.cfg.save_every is None:
                        new_ckpts.append("latest")
                    elif (self.step + 1) % self.cfg.save_every == 0:
                        new_ckpts.append(f"step_{self.step + 1}")

                    if (
                        self.ckpt_last_value is not None
                        and self.ckpt_last_value < self.ckpt_best_value
                    ):
                        new_ckpts.append("best")
                        self.log(f"saving best @ {self.step + 1} steps done")
                        self.ckpt_best_value = self.ckpt_last_value

                self.step += 1
                if len(new_ckpts) > 0:
                    self.save_checkpoint(new_ckpts)
                progress_bar.update(1)

    def save_checkpoint(self, names: str | list[str]):
        if self.cfg.run_dir is None:
            return

        main_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.random.get_rng_state(self.device)
        rng_state = torch.cat([main_rng_state, cuda_rng_state]).to(self.device)
        if self.world_size > 1:
            if self.is_main_process:
                gather_list = [
                    torch.zeros_like(rng_state) for _ in range(self.world_size)
                ]
            else:
                gather_list = None
            dist.gather(rng_state, gather_list, dst=0)
        else:
            gather_list = [rng_state]
        if not self.is_main_process:
            return
        ckpt = dict(
            step=self.step,
            epoch=self.epoch,
            # best_valid_loss=self.best_valid_loss,
            train_loader=self.train_loader.state_dict(),
            model=self.raw_model.state_dict(),
            muon=self.muon.state_dict(),
            adamw=self.adamw.state_dict(),
            rng=torch.stack(gather_list),
        )

        if isinstance(names, str):
            names = [names]
        for name in names:
            savepath = Path(self.cfg.run_dir) / f"{name}.ckpt.new"
            torch.save(ckpt, savepath)
            savepath.replace(Path(self.cfg.run_dir) / f"{name}.ckpt")

    def load_checkpoint(self, filepath: str):
        ckpt = torch.load(filepath)
        self.step = ckpt["step"]
        self.epoch = ckpt["epoch"]
        # self.best_valid_loss = ckpt["best_valid_loss"]
        self.train_loader.load_state_dict(ckpt["train_loader"])
        self.raw_model.load_state_dict(ckpt["model"])
        self.muon.load_state_dict(ckpt["muon"])
        self.adamw.load_state_dict(ckpt["adamw"])

        main_rng_size = torch.random.get_rng_state().size(0)
        torch.random.set_rng_state(ckpt["rng"][self.rank, :main_rng_size].cpu())
        torch.cuda.random.set_rng_state(ckpt["rng"][self.rank, main_rng_size:].cpu())
