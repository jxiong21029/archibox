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
fig.suptitle("Image RoPE, dot product with equivalent key to image center")

min_freq = torch.pi
max_freq = 50.0 * torch.pi

for row, dim in enumerate((128, 512)):
    inputs = torch.randn(b, 1, 1, dim)
    # Option 1: sin(w1 x), cos(w1 x), ..., sin(wk x), cos(wk x), ..., sin(w1 y), cos(w1 y),
    # ..., sin(wk y), cos(wk y)

    x1, x2, y1, y2 = inputs.chunk(4, dim=-1)
    freqs = min_freq * ((max_freq / min_freq) ** torch.linspace(0, 1, dim // 4))
    output_bnnd = torch.cat(
        [
            x1 * torch.cos(w * freqs) - x2 * torch.sin(w * freqs),
            x1 * torch.sin(w * freqs) + x2 * torch.cos(w * freqs),
            y1 * torch.cos(h * freqs) - y2 * torch.sin(h * freqs),
            y1 * torch.sin(h * freqs) + y2 * torch.cos(h * freqs),
        ],
        dim=-1,
    )
    assert output_bnnd.shape == (b, n, n, dim), f"{output_bnnd.shape=}"

    alignments = (output_bnnd * inputs).sum(dim=-1).mean(dim=0) / dim
    axes[row, 0].set_title(f"Axial RoPE, {dim=}")
    axes[row, 0].imshow(alignments, cmap="viridis", vmin=-1.0, vmax=1.0)

    # # Option 2: sin(w1 x), cos(w1 x), ..., sin(wk x), cos(wk x), ..., sin(w1 y), cos(w1 y),
    # # ..., sin(wk y), cos(wk y)

    # x1, x2 = inputs.chunk(2, dim=-1)

    # fw = (
    #     (torch.rand(dim // 2) * 2 - 1) * base * (mult ** torch.linspace(0, 1, dim // 2))
    # )
    # fh = (
    #     (torch.rand(dim // 2) * 2 - 1) * base * (mult ** torch.linspace(0, 1, dim // 2))
    # )
    # output_bnnd = torch.cat(
    #     [
    #         x1 * torch.cos(w * fw + h * fh) - x2 * torch.sin(w * fw + h * fh),
    #         x1 * torch.sin(w * fw + h * fh) + x2 * torch.cos(w * fw + h * fh),
    #     ],
    #     dim=-1,
    # )
    # assert output_bnnd.shape == (b, n, n, dim), f"{output_bnnd.shape=}"

    # alignments = (output_bnnd * inputs).sum(dim=-1).mean(dim=0) / dim
    # axes[row, 1].set_title(f"Mixed (independent) RoPE, {dim=}")
    # img = axes[row, 1].imshow(alignments, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    # # fig.colorbar(img, fraction=0.046, pad=0.04)

    # Option 3: sin(w1 x), cos(w1 x), ..., sin(wk x), cos(wk x), ..., sin(w1 y), cos(w1 y),
    # ..., sin(wk y), cos(wk y)

    x1, x2 = inputs.chunk(2, dim=-1)

    phi = torch.arange(dim // 2) * torch.pi * (-1 + 5**0.5)
    freqs = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    # freqs = torch.randn(dim // 2, 2)
    freqs = freqs / (freqs.pow(2).mean(dim=1, keepdim=True) + 1e-7).sqrt()
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
    axes[row, 1].set_title(f"Uniformly rotated RoPE, {dim=}")
    img = axes[row, 1].imshow(alignments, cmap="viridis", vmin=-1.0, vmax=1.0)
    fig.colorbar(img, fraction=0.046, pad=0.04)

fig.savefig("alignments_pos2d.png")
