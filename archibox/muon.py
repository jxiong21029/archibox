import torch

ortho_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)


@torch.compile
def orthogonalize(G):
    """Computes the semi-orthogonalization of G with Newton-Schulz iteration."""
    assert G.ndim >= 2
    X = G.type(ortho_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Coefficients from https://leloykun.github.io/ponder/muon-opt-coeffs/
    A = X @ X.mT
    X = 4.0848 * X + (-6.8946 * A + 2.9270 * A @ A) @ X
    A = X @ X.mT
    X = 3.9505 * X + (-6.3029 * A + 2.6377 * A @ A) @ X
    A = X @ X.mT
    X = 3.7418 * X + (-5.5913 * A + 2.3037 * A @ A) @ X
    A = X @ X.mT
    X = 2.8769 * X + (-3.1427 * A + 1.2046 * A @ A) @ X
    A = X @ X.mT
    X = 2.8366 * X + (-3.0525 * A + 1.2012 * A @ A) @ X

    if G.size(-2) > G.size(-1):
        # Scale to ensure that the norm of the ROWS of G (i.e. change in output) is 1
        X = X.mT * (G.size(-2) / G.size(-1)) ** 0.5
    return X.type_as(G)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    See: https://kellerjordan.github.io/posts/muon/, https://arxiv.org/abs/2409.20325

    NOTE: This optimizer should not be used for the embedding layer, the final fully
    connected layer, or any {0,1}-D parameters; those should be optimized by a standard
    method (e.g., AdamW).

    NOTE: This is a naive implementation in which every device computes the
    orthogonalization for all parameters. Compatible with DistributedDataParallel.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                g = param.grad
                if g is None:
                    continue

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = orthogonalize(g).add_(param, alpha=weight_decay)
                param.sub_(g, alpha=lr)


# class DistributedMuon(torch.optim.Optimizer):
#     def __init__(
#         self,
#         params,
#         lr: float = 0.01,
#         momentum: float = 0.9,
#         weight_decay: float = 0.01,
#         nesterov: bool = True,
#     ):
#         defaults = dict(
#             lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
#         )
#         super().__init__(params, defaults)
#         self.rank = int(os.environ["RANK"])
#         self.world_size = int(os.environ["WORLD_SIZE"])

#         for group in self.param_groups:
#             group["ordered_params"] = sorted(group["params"], key=lambda p: p.numel())

#     @torch.no_grad
#     def step(self):
#         for group in self.param_groups:
#             lr = group["lr"]
#             weight_decay = group["weight_decay"]
#             momentum = group["momentum"]
#             nesterov = group["nesterov"]

#             responsible_rank = 0
#             last_reduce = None
#             last_sync = None
#             queue = [None for _ in range(self.world_size)]

#             for param in group["ordered_params"]:
#                 grad = param.grad
#                 if grad is None:
#                     continue

#                 # Divide then sum to avoid overflow
#                 grad.div_(self.world_size)
#                 reduce_handle = dist.reduce(grad, dst=responsible_rank, async_op=True)
#                 if self.rank == responsible_rank:
#                     state = self.state[param]
#                     if "momentum_buffer" not in state:
#                         state["momentum_buffer"] = torch.zeros_like(grad)
#                     buf = state["momentum_buffer"]
#                     buf.lerp_(grad, 1 - momentum)
#                     grad = grad.lerp_(buf, momentum) if nesterov else buf
#                     grad = orthogonalize(grad).add_(param, alpha=weight_decay)
#                 sync_handle = dist.broadcast(param, src=responsible_rank, async_op=True)

#                 # if queue[responsible_rank] is not None:
#                 #     if self.rank == responsible_rank:
#                 #         last_update.wait()
#                 #         queue[responsible_rank].sub_(
#                 #             queue[responsible_rank].grad, alpha=lr
#                 #         )
#                 #     sync_handle = dist.broadcast(
#                 #         queue[responsible_rank], src=responsible_rank, async_op=True
#                 #     )
#                 #     if self.rank == responsible_rank:
#                 #         if last_sync is not None:
#                 #             last_sync.wait()
#                 #         last_sync = sync_handle
#                 # queue[responsible_rank] = param

#                 # if self.rank == responsible_rank:

#                 #     def compute_update(fut):
#                 #         g = fut.value()[0]
#                 #         state = self.state[param]
#                 #         if "momentum_buffer" not in state:
#                 #             state["momentum_buffer"] = torch.zeros_like(g)
#                 #         buf = state["momentum_buffer"]
#                 #         buf.lerp_(g, 1 - momentum)
#                 #         g = g.lerp_(buf, momentum) if nesterov else buf
#                 #         g = orthogonalize(g).add_(param, alpha=weight_decay)
#                 #         grad.copy_(g)
#                 #         return grad

#                 #     last_update = reduce_handle.get_future().then(compute_update)

#                 responsible_rank = (responsible_rank + 1) % self.world_size

#             if last_update is not None:
#                 last_update.wait()
#             if last_sync is not None:
#                 last_sync.wait()

#             for i, param in enumerate(queue):
#                 if param is not None:
#                     dist.broadcast(queue[i], src=i)
