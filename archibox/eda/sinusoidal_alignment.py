from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def main():
    matplotlib.use("agg")
    sns.set_theme()

    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True)
    fig.set_size_inches(16, 16)

    for i in range(len(axes)):
        for j in range(len(axes[0])):
            # min_freq = 2 * i + 1
            min_freq = torch.pi
            max_mult = [50, 100, 200][j]

            freqs = min_freq * (max_mult ** torch.linspace(0, 1, 128 * (2**i)))

            xx = torch.linspace(0, 1, 51)
            x_emb = xx.unsqueeze(1) * freqs
            z_ND = torch.cat([x_emb.sin(), x_emb.cos()], dim=1)

            x0_emb = torch.zeros_like(freqs)
            z0_D = torch.cat([x0_emb.sin(), x0_emb.cos()], dim=0)

            alignments = 2.0 * (z_ND * z0_D).mean(dim=-1)

            axes[i, j].set_title(
                f"minf={min_freq:.1f}, maxm={max_mult:.1e}, n={len(freqs)}, "
                f"v={freqs.pow(2).mean().sqrt():.1e}, "
                f"i={min(i for i in range(len(alignments)) if alignments[i] < 1024**-0.5)}"
            )
            axes[i, j].plot(xx, alignments)
            axes[i, j].axhline(1024**-0.5, c="gray", ls="--", alpha=0.8)
            axes[i, j].axhline(-(1024**-0.5), c="gray", ls="--", alpha=0.8)

    fig.savefig(Path(__file__).parent / "out.jpg")


if __name__ == "__main__":
    main()
