import math

import numpy as np
import torch


def uniform_directions(N: int, D: int) -> torch.Tensor:
    # Quasi-random samples from U[0, 1] using Kronecker sequences
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    x = 23
    while len(primes) < D:
        if all(x % p != 0 for p in primes):
            primes.append(x)
        x += 2
    z = torch.outer(
        torch.arange(1, N + 1),
        torch.sqrt(torch.tensor(primes[:D], dtype=torch.float64)),
    ).fmod(1.0)

    # Map from U[0, 1] to N(0, 1)
    z = math.sqrt(2) * torch.erfinv(2 * z - 1)

    directions = z / z.norm(dim=1, keepdim=True)
    return directions.float()


def main():
    N = 64
    D = 3

    torch.manual_seed(0)

    unif_directions = uniform_directions(N, D)
    assert unif_directions.shape == (N, D)
    print(f"{unif_directions.std()=:.8f}")

    # --- Test: distance of mean from 0 ---

    # rand_offsets = []
    # for _ in range(100):
    #     z = torch.randn(N, D)
    #     rand_directions = z / z.norm(dim=1, keepdim=True)
    #     rand_offsets.append(rand_directions.mean(dim=0).pow(2).mean().item())
    # print(np.mean(rand_offsets))

    # unif_directions = uniform_directions(N, D)
    # print(unif_directions.mean(dim=0).pow(2).mean())

    # --- Test: cosine alignment thresholding ---
    rand_results = []
    unif_results = []
    for _ in range(1000):
        z = torch.randn(N, D)
        rand_directions = z / z.norm(dim=1, keepdim=True)

        query = torch.randn(D)
        rand_prop = ((rand_directions * query).sum(dim=1) > 0.5).float().mean().item()
        unif_prop = ((unif_directions * query).sum(dim=1) > 0.5).float().mean().item()
        rand_results.append(rand_prop)
        unif_results.append(unif_prop)
    print(np.std(rand_results), np.std(unif_results))


if __name__ == "__main__":
    main()
