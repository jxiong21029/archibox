"""
Can RoPE output arbitrary distributions...?
"""

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

matplotlib.use("agg")


def extract_patches(
    images_NCHW: torch.Tensor,
    y0_NT: torch.Tensor,
    x0_NT: torch.Tensor,
    y1_NT: torch.Tensor,
    x1_NT: torch.Tensor,
    P: int,
):
    """Extracts patches from batch of images using area-style intepolation.

    T: number of patches per image
    P: resized patch side length, in pixels
    """
    assert torch.is_floating_point(images_NCHW)
    N, C, H, W = images_NCHW.shape
    _, T = y0_NT.shape
    assert y0_NT.shape == y1_NT.shape == x0_NT.shape == x1_NT.shape == (N, T)

    offsets = torch.linspace(0, 1, P + 1, device=images_NCHW.device)
    ys_NTP = y0_NT[..., None] + (y1_NT - y0_NT)[..., None] * offsets
    xs_NTP = x0_NT[..., None] + (x1_NT - x0_NT)[..., None] * offsets
    ys_NTPP = ys_NTP.view(N, T, P + 1, 1).expand(N, T, P + 1, P + 1)
    xs_NTPP = xs_NTP.view(N, T, 1, P + 1).expand(N, T, P + 1, P + 1)
    grid_xy_NTPP2 = torch.stack([xs_NTPP, ys_NTPP], dim=-1)

    integral = torch.zeros((N, C, H + 1, W + 1), device=images_NCHW.device)
    integral[:, :, 1:, 1:] = images_NCHW.cumsum(dim=-1).cumsum(dim=-2)

    integral_samples = F.grid_sample(
        einops.repeat(integral, "N C H W -> (N T) C H W", T=T),
        grid_xy_NTPP2.reshape(N * T, P + 1, P + 1, 2).float(),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).reshape(N, T, C, P + 1, P + 1)

    i00 = integral_samples[..., :-1, :-1]
    i01 = integral_samples[..., :-1, 1:]
    i10 = integral_samples[..., 1:, :-1]
    i11 = integral_samples[..., 1:, 1:]

    # Box-sum via 4-corner trick, then divide by pixel area
    sums = i11 - i01 - i10 + i00
    area = (y1_NT - y0_NT) * H / 2 / P * (x1_NT - x0_NT) * W / 2 / P
    patches = sums / area.view(N, T, 1, 1, 1)
    return patches.type_as(images_NCHW)


class Model(nn.Module):
    def __init__(self, dim: int, min_freq: float, max_freq: float, H: int, W: int):
        super().__init__()
        self.q_D = nn.Parameter(torch.randn(dim))
        self.k_D = nn.Parameter(torch.randn(dim))

        n_freqs = dim // 2
        phi_F = torch.pi * (-1 + 5**0.5) * torch.arange(n_freqs)
        freqs_F = torch.stack([torch.cos(phi_F), torch.sin(phi_F)], dim=-1)
        freqs_FP = (
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs)
        ).unsqueeze(-1) * freqs_F
        self.freqs_FP = nn.Buffer(freqs_FP, persistent=False)

        y = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
        x = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
        pos_HWP = torch.stack([y, x], dim=-1)
        theta_HWF = (self.freqs_FP * pos_HWP[..., None, :]).mean(dim=-1)
        self.cos_HWF = nn.Buffer(torch.cos(theta_HWF), persistent=False)
        self.sin_HWF = nn.Buffer(torch.sin(theta_HWF), persistent=False)

    def forward(self):
        x_F, y_F = self.k_D.chunk(2, dim=-1)
        x_out_HWF = x_F * self.cos_HWF - y_F * self.sin_HWF
        y_out_HWF = x_F * self.sin_HWF + y_F * self.cos_HWF
        result_HWD = torch.cat([x_out_HWF, y_out_HWF], dim=-1)
        sim_HW = (self.q_D * result_HWD).mean(dim=-1)
        return sim_HW

    def loss(self, target_HW):
        return (self() - target_HW).pow(2).mean()


resolution = 64
min_freq = 10.0
max_freq = 100_000.0


img = Image.open("archibox/lodim/peak.jpg")
data = np.array(img)
data = torch.tensor(data).movedim(-1, -3)
data = extract_patches(
    data.unsqueeze(0).float() / 255.0,
    torch.tensor([[-1.0]]),
    torch.tensor([[-1.0]]),
    torch.tensor([[1.0]]),
    torch.tensor([[1.0]]),
    resolution,
)[0, 0]
data_HW = data.mean(dim=0).cuda()

DIMS = [1024, 4096, 16384]
N_STEPS = [8_000, 4000, 2000]

fig, axes = plt.subplots(ncols=1 + len(DIMS), constrained_layout=True)
fig.set_size_inches(16, 5)
axes[0].set_title("Original image")
axes[0].imshow(data_HW.cpu().numpy(), origin="upper", cmap="gray")

for i, (dim, n_steps) in enumerate(zip(DIMS, N_STEPS)):
    model = Model(dim, min_freq, max_freq, resolution, resolution)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=2e-2)

    for step in range(n_steps):
        loss = model.loss(data_HW)
        print(f"{dim=} || {step=} || {loss=:.6f}")
        optim.zero_grad()
        loss.backward()
        optim.step()

    output = model()
    axes[i + 1].set_title(f"Learned attention scores, head_dim={dim}")
    axes[i + 1].imshow(output.detach().cpu().numpy(), origin="upper", cmap="gray")
fig.savefig("tmp.png")
