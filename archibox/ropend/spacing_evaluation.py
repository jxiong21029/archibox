import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.distributions import Categorical

matplotlib.use("agg")
device = "cuda"


def evaluate(direction_spacing: float):
    torch.manual_seed(0)

    N, H, W, d = 128, 99, 99, 128
    # direction_spacing = math.pi * (1 - math.sqrt(5))
    n_freqs = d // 2
    temperature = 5.0

    variances = []
    entropies = []

    for min_freq in tqdm.tqdm(
        torch.exp(torch.linspace(math.log(0.1), math.log(10.0), 50)).to(device)
    ):
        max_freq = min_freq * 100.0

        q_Nd = torch.randn(N, d).to(device)
        omega_F = min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs).to(
            device
        )
        phi_F = torch.arange(n_freqs).to(device) * direction_spacing
        directions_F2 = torch.stack((torch.cos(phi_F), torch.sin(phi_F)), dim=-1)
        freqs_F2 = omega_F[:, None] * directions_F2

        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_HW = torch.linspace(-xlim, xlim, W).view(1, W).expand(H, W).to(device)
        y_HW = torch.linspace(-ylim, ylim, H).view(H, 1).expand(H, W).to(device)
        positions_HW2 = torch.stack((x_HW, y_HW), dim=-1)
        theta_HWF = (positions_HW2[:, :, None, :] * freqs_F2).sum(dim=-1)

        cos_HWF = torch.cos(theta_HWF)
        sin_HWF = torch.sin(theta_HWF)
        qx_NF, qy_NF = q_Nd.chunk(2, dim=-1)
        kx_NHWF = cos_HWF * qx_NF[:, None, None, :] - sin_HWF * qy_NF[:, None, None, :]
        ky_NHWF = sin_HWF * qx_NF[:, None, None, :] + cos_HWF * qy_NF[:, None, None, :]
        k_NHWd = torch.cat((kx_NHWF, ky_NHWF), dim=-1)

        alignments_HW = (q_Nd[:, None, None, :] * k_NHWd).mean(dim=(0, -1))
        density_HW = torch.softmax(
            alignments_HW.flatten() / temperature, dim=0
        ).reshape(H, W)

        assert torch.isclose(density_HW.sum(), torch.tensor(1.0))

        distances_HW = positions_HW2.pow(2).sum(dim=-1)
        variance = (distances_HW * density_HW).sum().item()
        entropy = Categorical(density_HW.flatten()).entropy().item()
        variances.append(variance)
        entropies.append(entropy)

    return variances, entropies

    # fig, ax = plt.subplots()
    # img = ax.imshow(alignments_HW.numpy(), cmap="viridis", vmin=-0.25, vmax=1.0)
    # fig.colorbar(img, fraction=0.046, pad=0.04)
    # fig.savefig(Path(__file__).parent / "figures" / "spacing_eval.png")


def main():
    fig, ax = plt.subplots()

    for direction_spacing, name in (
        (math.pi / 2, "axial"),
        (math.pi * (math.sqrt(5) - 1) / 2, "pi / golden"),
        (math.pi * (math.sqrt(5) - 1), "2pi / golden"),
    ):
        variances, entropies = evaluate(direction_spacing)
        ax.plot(variances, entropies, label=name)
    ax.set_xlabel("variance")
    ax.set_ylabel("entropy")
    fig.legend()
    fig.savefig(Path(__file__).parent / "figures" / "spacing_eval.png")


if __name__ == "__main__":
    main()
