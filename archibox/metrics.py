import logging
from collections import defaultdict

import torch
import wandb

log = logging.getLogger(__name__)


class Metrics:
    def __init__(self, enabled: bool, use_wandb: bool = True):
        self.enabled = enabled
        self.use_wandb = use_wandb

        self.context = ""
        self.n = defaultdict(int)
        self.mean = defaultdict(float)

    @torch.no_grad
    def push(self, **metrics):
        if not self.enabled:
            return

        for k, v in metrics.items():
            ck = self.context + k
            if isinstance(v, torch.Tensor):
                assert v.numel() == 1, f"{k} shape={v.shape}"
                delta = v.float().detach().view(()) - self.mean[ck]
            else:
                delta = float(v) - self.mean[ck]
            self.n[ck] += 1
            self.mean[ck] += delta / self.n[ck]

    def report(self):
        """averages metrics and logs to wandb"""
        if not self.enabled:
            return

        results = {}
        for k in self.n:
            if self.n[k] == 0:
                continue
            if isinstance(self.mean[k], torch.Tensor):
                results[k] = self.mean[k].to("cpu", non_blocking=True)
            else:
                results[k] = self.mean[k]
        torch.cuda.synchronize()
        results = {k: float(v) for k, v in results.items()}

        if self.use_wandb:
            wandb.log(results)
        else:
            log.info(", ".join(f"{k}: {v:.4g}" for k, v in results.items()))

        for k in self.n:
            self.n[k] = 0
            if isinstance(self.mean[k], torch.Tensor):
                self.mean[k].zero_()
            else:
                self.mean[k] = 0.0


@torch.no_grad
def rms(x, dim=-1):
    return x.float().pow(2).mean(dim=dim).sqrt().mean()
