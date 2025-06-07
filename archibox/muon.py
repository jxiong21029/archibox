import logging

import torch
import torch.nn as nn

ortho_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)

log = logging.getLogger(__name__)


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


def split_muon_adamw_params(
    module: nn.Module,
    verbose: bool = False,
) -> tuple[
    list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]
]:
    muon_params = []
    scalar_params = []
    embeds_params = []
    output_params = []
    for name, p in module.named_parameters():
        shape = tuple(p.shape)
        if not p.requires_grad:
            msg = f"{name} {shape} requires_grad=False, skipped"
        elif p.ndim < 2 or (hasattr(p, "_is_scalar") and p._is_scalar):
            msg = f"{name} {shape} (scalar) assigned to AdamW"
            scalar_params.append(p)
        elif hasattr(p, "_is_embed") and p._is_embed:
            msg = f"{name} {shape} (embed) assigned to AdamW"
            embeds_params.append(p)
        elif hasattr(p, "_is_output") and p._is_output:
            msg = f"{name} {shape} (output) assigned to AdamW"
            output_params.append(p)
        else:
            if hasattr(p, "_ortho"):
                raise ValueError(
                    "_ortho is deprecated, use _is_embed or _is_output instead"
                )
            msg = f"{name} {shape} assigned to Muon"
            muon_params.append(p)
        if verbose:
            log.info(msg)

    if verbose:
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
    return muon_params, scalar_params, embeds_params, output_params
