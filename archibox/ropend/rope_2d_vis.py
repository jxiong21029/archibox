import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("agg")

torch.manual_seed(1)

b = 16
n = 64
h = torch.linspace(-1, 1, n).view(1, n, 1).expand(n, n, 1)
w = torch.linspace(-1, 1, n).view(n, 1, 1).expand(n, n, 1)
if n % 2 == 0:
    h = h + 1 / (n - 1)
    w = w + 1 / (n - 1)

fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
fig.set_size_inches(12, 12)
fig.suptitle(
    "Image RoPE, dot product of query with the rotation of that query over varying positions"
)

min_freq = torch.pi
max_freq = 50.0 * torch.pi


def maximin(x_ND: torch.Tensor):
    N, D = x_ND.shape
    idx = torch.zeros(N, dtype=torch.long)
    lengths = torch.zeros(N)

    k = 0
    dists = (x_ND - x_ND[k]).pow(2).sum(dim=-1)
    idx[-1] = k
    lengths[-1] = torch.inf
    start = N - 2

    for i in range(start, -1, -1):
        k = torch.argmax(dists)
        idx[i] = k
        lengths[i] = dists[k]
        dists = torch.minimum(dists, (x_ND - x_ND[k]).pow(2).sum(dim=-1))

    return idx, lengths


# phi_F = torch.arange(10) * torch.pi * (1 - 5**0.5)
# x_F2 = torch.stack((torch.cos(phi_F), torch.sin(phi_F)), dim=-1)
# print(maximin(x_F2))


for row, dim in enumerate((128, 512)):
    inputs = torch.randn(b, 1, 1, dim)

    # ------ AXIAL --------

    # x1, x2, y1, y2 = inputs.chunk(4, dim=-1)
    # freqs = min_freq * ((max_freq / min_freq) ** torch.linspace(0, 1, dim // 4))
    # output_bnnd = torch.cat(
    #     [
    #         x1 * torch.cos(w * freqs) - x2 * torch.sin(w * freqs),
    #         x1 * torch.sin(w * freqs) + x2 * torch.cos(w * freqs),
    #         y1 * torch.cos(h * freqs) - y2 * torch.sin(h * freqs),
    #         y1 * torch.sin(h * freqs) + y2 * torch.cos(h * freqs),
    #     ],
    #     dim=-1,
    # )
    # assert output_bnnd.shape == (b, n, n, dim), f"{output_bnnd.shape=}"

    # alignments = (output_bnnd * inputs).sum(dim=-1).mean(dim=0) / dim
    # axes[row, 0].set_title(f"Axial RoPE, {dim=}")
    # axes[row, 0].imshow(alignments, cmap="viridis", vmin=-1.0, vmax=1.0)

    # ------- UNIFORM ----------

    x1, x2 = inputs.chunk(2, dim=-1)

    n_freqs = dim // 2
    phi = torch.arange(n_freqs) * torch.pi * (-1 + 5**0.5)
    freqs = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    freqs = freqs * (
        min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, dim // 2)
    ).unsqueeze(-1)  # dim // 2, 2
    pos = torch.cat([h, w], dim=-1)  # n n 2
    theta = (freqs * pos.unsqueeze(-2)).mean(dim=-1)

    output_bnnd = torch.cat(
        [
            x1 * torch.cos(theta) - x2 * torch.sin(theta),
            x1 * torch.sin(theta) + x2 * torch.cos(theta),
        ],
        dim=-1,
    )
    assert output_bnnd.shape == (b, n, n, dim), f"{output_bnnd.shape=}"

    alignments = (output_bnnd * inputs).sum(dim=-1).mean(dim=0) / dim
    alignments = alignments.numpy()
    axes[row, 0].set_title(f"Uniformly rotated RoPE, {dim=}")
    img = axes[row, 0].imshow(alignments, cmap="viridis", vmin=-1.0, vmax=1.0)

    x1, x2 = inputs.chunk(2, dim=-1)
    phi = torch.arange(n_freqs) * torch.pi * (-1 + 5**0.5)
    freqs = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    idx, _ = maximin(freqs)
    # freqs = freqs[idx]
    freqs = freqs[torch.flip(idx, dims=(0,))]
    freqs = freqs * (
        min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, dim // 2)
    ).unsqueeze(-1)
    theta = (freqs * pos.unsqueeze(-2)).mean(dim=-1)

    output_bnnd = torch.cat(
        [
            x1 * torch.cos(theta) - x2 * torch.sin(theta),
            x1 * torch.sin(theta) + x2 * torch.cos(theta),
        ],
        dim=-1,
    )
    alignments = (output_bnnd * inputs).sum(dim=-1).mean(dim=0) / dim
    alignments = alignments.numpy()
    axes[row, 1].set_title(f"Minimax rotated RoPE, {dim=}")
    img = axes[row, 1].imshow(alignments, cmap="viridis", vmin=-1.0, vmax=1.0)

    fig.colorbar(img, fraction=0.046, pad=0.04)

fig.savefig("alignments_pos2d.png")
