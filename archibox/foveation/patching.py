import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bnnp.nn import FusedLinear
from einops import rearrange
from torch import Tensor

from archibox.foveation.multirotary import MultiRotaryDecoder


class FourierEmbed(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nfreqs: int,
        min_freq: float,
        max_freq: float,
        frozen: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim

        z = torch.randn(nfreqs, in_dim)
        z = z / (z.pow(2).mean(dim=1, keepdim=True) + 1e-7).sqrt()
        self.z = nn.Parameter(z, requires_grad=not frozen)
        self.freqs_H = nn.Parameter(
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, nfreqs),
            requires_grad=not frozen,
        )

    def forward(self, x_ND):
        assert x_ND.size(-1) == self.in_dim
        alignments_NH = (self.z * x_ND.unsqueeze(-2)).mean(dim=-1)
        theta_NH = self.freqs_H * alignments_NH
        return torch.cat([torch.sin(theta_NH), torch.cos(theta_NH)], dim=-1)


class PatchDecoder(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        dim: int,
        mlp_dim: int,
        depth: int,
        nfreqs: int,
        seq_len: int,
    ):
        super().__init__()
        self.patchify = FusedLinear(np.prod(patch_size), dim)

        # Include both absolute and relative positional embedding
        self.pos_embed = FourierEmbed(3, nfreqs, min_freq=0.1, max_freq=100.0)
        self.emb_proj = FusedLinear(nfreqs * 2, dim)

        self.decoder = MultiRotaryDecoder(
            dim,
            mlp_dim,
            head_dim=64,
            pos_dim=3,
            depth=depth,
            seq_len=seq_len,
            use_rope=True,
            min_freq=0.1,
            max_freq=100.0,
            frozen_rope=True,
        )

    def forward(self, patches_NTCHW, pos_NT2, sizes_NT):
        """
        Args:
            patches_NTHWC: patches (should be already resized) extracted from images
            pos_NT2: coordinates of patch centers in [-1, 1].
                -1 is bottom/left, +1 is top/right
            sizes_NT: sizes of patches in [0, 1]
                0 is size 0, 1 is full image
        """
        assert torch.is_floating_point(patches_NTCHW)
        N, T, _, _, _ = patches_NTCHW.shape

        patches_NTD = rearrange(patches_NTCHW, "N T C H W -> N T (C H W)")
        x_NTD = self.patchify(patches_NTD)

        pos_NT3 = torch.cat([pos_NT2, sizes_NT.unsqueeze(-1)], dim=-1)

        pos_emb = self.pos_embed(pos_NT3).type_as(x_NTD)
        emb_NTD = self.emb_proj(pos_emb)
        return self.decoder(x_NTD + emb_NTD, pos_NT3)


def build_patch_locs(nlevels: int = 4, overlap: bool = True) -> Tensor:
    """Returns a tensor of shape (total_patches, 4) containing (y_min, x_min, y_max,
    x_max) for every patch in every level.

    Coordinates are in [-1, 1].
    """
    locs = []

    for level in range(nlevels):
        n = 2 ** (level + 1) - 1 if overlap else 2**level
        half_side = 1 / 2**level
        centers = torch.linspace(-1, 1, n + 2)[1:-1]

        cy, cx = torch.meshgrid(centers, centers, indexing="ij")  # (n, n)
        new_locs = torch.stack(
            [cy - half_side, cx - half_side, cy + half_side, cx + half_side], dim=-1
        )  # (n, n, 4)
        locs.append(new_locs.reshape(-1, 4))  # flatten to (n^2, 4)

    return torch.cat(locs, dim=0)


def extract_patches(
    images_NCHW: Tensor,
    y0_NT: Tensor,
    x0_NT: Tensor,
    y1_NT: Tensor,
    x1_NT: Tensor,
    P: int,
):
    """Extracts patches from batch of images using area-style intepolation.

    T: number of patches per image
    P: resized patch side length, in pixels

    Returns patches of shape (N, T, C, P, P)
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


def visualize_patches(
    image_CHW: Tensor,
    label: int,
    class_logits: Tensor,
    patch_locs: Tensor,
    patch_idx: Tensor,
    P: int,
):
    import matplotlib.pyplot as plt

    C, H, W = image_CHW.shape
    probs = F.softmax(class_logits, dim=-1).cpu().numpy()

    frames = []
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(10, 10)
    axes[0, 0].set_title("Full image")
    axes[0, 0].imshow(
        image_CHW.movedim(-3, -1).float().cpu().numpy(),
        origin="upper",
        cmap="gray" if C == 1 else None,
    )
    axes[0, 0].axis("off")

    axes[0, 1].set_title("Model views")
    axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].set_ylim(1, -1)
    axes[0, 1].axis("off")

    T = probs.shape[0]
    patch_locs_T4 = patch_locs[patch_idx]
    patches = extract_patches(
        image_CHW.unsqueeze(0).float() / 255.0,
        patch_locs_T4[:, 0].view(1, T),
        patch_locs_T4[:, 1].view(1, T),
        patch_locs_T4[:, 2].view(1, T),
        patch_locs_T4[:, 3].view(1, T),
        P,
    ).cpu()
    assert patches.shape == (1, T, C, P, P)
    patches = patches.squeeze(0)
    patches = patches.clamp(0, 1).mul(255).type(torch.uint8)
    patches = patches.movedim(-3, -1).cpu().numpy()
    patch_locs_T4 = patch_locs_T4.cpu().numpy()

    for t in range(probs.shape[0]):
        y0, x0, y1, x1 = patch_locs_T4[t]
        axes[0, 1].imshow(
            patches[t],
            origin="upper",
            extent=(x0, x1, y1, y0),
            cmap="gray" if C == 1 else None,
        )

        # axes[1, 0].clear()
        # axes[1, 0].set_title("Class distribution")
        # axes[1, 0].barh(np.arange(10), probs[t])
        # axes[1, 0].set_xlim(0, 1.05)
        # axes[1, 0].set_ylim(10, -1)
        # axes[1, 0].set_yticks(np.arange(10))

        axes[1, 1].clear()
        axes[1, 1].set_title(f"P(label == {int(label)})")
        axes[1, 1].set_xlim(0, probs.shape[0])
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].plot(probs[: t + 1, int(label)])

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3])
    plt.close(fig)

    return frames
