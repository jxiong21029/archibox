import torch


def fourier_embed(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding for time. Expects input of shape (..., 1)"""
    assert t.size(-1) == 1
    thetas = t.float() * freqs.float()
    return torch.cat([thetas.sin(), thetas.cos()], dim=-1)


def default_freqs(K: int) -> torch.Tensor:
    """Half-cycle base followed by integer multiples."""
    f0 = torch.pi / 2
    return f0 * torch.arange(1, K + 1)


def grid_search(y, freqs, n_samples=400):
    t_grid = torch.linspace(-1.0, 1.0, n_samples + 2, device=y.device)[1:-1]
    embed = torch.cat(
        [
            torch.sin(freqs[None] * t_grid[:, None]),
            torch.cos(freqs[None] * t_grid[:, None]),
        ],
        dim=-1,
    )
    err = (y.unsqueeze(-2) - embed).pow(2).mean(dim=-1)
    idx = err.argmin(dim=-1)
    return t_grid[idx]


def refine(t0, y, freqs, n_iter=20):
    # s = (1 - torch.exp(-t0)) / (1 + torch.exp(-t0))
    s = torch.log((1 + t0) / (1 - t0))
    s = s.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([s], max_iter=n_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = (1 - torch.exp(-s)) / (1 + torch.exp(-s))
        pred = torch.cat(
            [torch.sin(freqs * t[..., None]), torch.cos(freqs * t[..., None])], -1
        )
        loss = (pred - y).pow(2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    t = (1 - torch.exp(-s)) / (1 + torch.exp(-s))
    return t.detach()


def recover_t(y: torch.Tensor, K: int = 6, n_grid: int = 400) -> torch.Tensor:
    freqs = default_freqs(K).to(y.device)
    t0 = grid_search(y, freqs, n_grid)
    return refine(t0, y, freqs)


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("agg")

    fig, ax = plt.subplots()
    ax.plot(
        np.arange(512), np.sort(recover_t(torch.randn(512, 12).cuda()).cpu().numpy())
    )
    ax.plot(
        np.arange(512), np.sort(recover_t(torch.randn(512, 12).cuda()).cpu().numpy())
    )
    ax.plot(
        np.arange(512), np.sort(recover_t(torch.randn(512, 12).cuda()).cpu().numpy())
    )
    ax.plot(np.arange(512), np.linspace(-1, 1, 512), c="black", ls="--")
    fig.savefig("tmp.png")


if __name__ == "__main__":
    main()
