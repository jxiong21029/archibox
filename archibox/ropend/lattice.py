import math

import torch


def uniform_directions(N: int, D: int) -> torch.Tensor:
    # Quasi-random samples from the uniform distribution using Kronecker sequences
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    x = 23
    while len(primes) < D:
        if all(x % p != 0 for p in primes):
            primes.append(x)
        x += 2
    z = torch.outer(
        torch.arange(1, N + 1),
        torch.sqrt(torch.tensor(primes[:D], dtype=torch.float64)),
    )

    # Maps samples from U[0, 1] to N(0, 1)
    z = math.sqrt(2) * torch.erfinv(2 * z.fmod(1.0) - 1)

    directions = z / z.norm(dim=1, keepdim=True)
    return directions.float()


def main():
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    xs, ys, zs = uniform_directions(N=32, D=3).unbind(-1)

    ax.scatter(xs.numpy().tolist(), ys.numpy().tolist(), zs.numpy().tolist())
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    fig.savefig(Path(__file__).parent / "figures/lattice.png")


if __name__ == "__main__":
    main()
