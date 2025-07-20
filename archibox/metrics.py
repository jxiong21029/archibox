import logging
import time
from collections import defaultdict, deque

import torch

log = logging.getLogger(__name__)


@torch.no_grad()
def rms(x, dim=-1):
    return x.float().pow(2).mean(dim=dim).sqrt().mean()


class Metrics:
    def __init__(
        self,
        enabled: bool = True,
        use_wandb: bool = False,
        use_cuda_events: bool = False,
    ):
        self.enabled = enabled
        self.use_wandb = use_wandb

        self.context = ""
        self.n = defaultdict(int)
        self.mean = defaultdict(float)

        self.timed_events = deque(maxlen=1024)
        self.last_t = None
        self.curr_event = None

    @torch.no_grad()
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

    def tick(self, name: str | None):
        if self.curr_event is not None:
            if self.use_cuda_events:
                now = torch.cuda.Event(enable_timing=True)
                now.record()
            else:
                now = time.perf_counter()
            self.timed_events.append(
                (self.curr_event, self.last_t, time.perf_counter())
            )

        self.curr_event = name
        if self.use_cuda_events:
            self.last_t = torch.cuda.Event(enable_timing=True)
            self.last_t.record()
        else:
            self.last_t = time.perf_counter()

    def tock(self):
        if self.use_cuda_events and len(self.timed_events) > 0:
            torch.cuda.synchronize()
        for k, start, end in self.timed_events:
            if self.use_cuda_events:
                elapsed = start.elapsed_time(end) / 1000.0
            else:
                elapsed = end - start
            self.metrics.push(**{f"{k}_sec": elapsed})
        self.timed_events.clear()

    def report(self):
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
            import wandb

            wandb.log(results)
        else:
            log.info(", ".join(f"{k}: {v:.4g}" for k, v in results.items()))

        for k in self.n:
            self.n[k] = 0
            if isinstance(self.mean[k], torch.Tensor):
                self.mean[k].zero_()
            else:
                self.mean[k] = 0.0
