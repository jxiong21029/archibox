import torch


def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x


def uniform_directions(n: int, d: int) -> torch.Tensor:
    g = _phi(d)
    alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return directions.float()


def main():
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    xs, ys, zs = uniform_directions(n=4096, d=3).unbind(-1)
    # directions = torch.randn(4096, 3)
    # directions = directions / directions.norm(dim=1, keepdim=True)
    # xs, ys, zs = directions.unbind(-1)

    ax.scatter(xs.numpy().tolist(), ys.numpy().tolist(), zs.numpy().tolist(), s=1)
    ax.set_box_aspect(
        (
            (xs.max() - xs.min()).item(),
            (ys.max() - ys.min()).item(),
            (zs.max() - zs.min()).item(),
        )
    )

    fig.savefig(Path(__file__).parent / "figures/lattice.png", dpi=500)


if __name__ == "__main__":
    main()
