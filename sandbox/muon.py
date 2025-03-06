import torch


@torch.compile
def newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt
    to use a quintic iteration whose coefficients are selected to maximize the slope at
    zero. For the purpose of minimizing steps, it turns out to be empirically effective
    to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration
    therefore does not produce UV^T but rather something like US'V^T where S' is
    diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X /= X.norm(dim=(-2, -1), keepdim=True) + 1e-7
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the
    nearest orthogonal matrix. To efficiently orthogonalize each update, we use a
    Newton-Schulz iteration, which has the advantage that it can be stably run in
    bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully
    connected layer, or any {0,1}-D parameters; those should all be optimized by a
    standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last
    3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for param in group["params"]:
                g = param.grad
                if g is None:
                    continue

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf = buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = newtonschulz5(g.reshape(g.size(0), -1), ns_steps)

                # Ensures expected norm of change in output doesn't depend on input dim
                g = g * max(1, g.size(-2) / g.size(-1)) ** 0.5
                # If out_dim := g.size(-2) <= in_dim := g.size(-1), then g has rms 1 /
                # sqrt(in_dim) and is unchanged, so the expected norm of the change in
                # output is equal to 1. If out_dim := g.size(-2) > in_dim := g.size(-1),
                # then g has rms 1 / sqrt(out_dim) * (sqrt(out_dim) / sqrt(in_dim)) = 1
                # / sqrt(in_dim), so the expected norm of the output change is still 1.

                param.sub_(g.view_as(param) + param * weight_decay, alpha=lr)
