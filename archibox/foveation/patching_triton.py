import einops
import torch
import torch.nn.functional as F
from torch import Tensor

# @triton.jit
# def _batched_grid_sample_kernel(
#     images,  # ptr to float32, shape (N, C, H, W)
#     grid,  # ptr to float32, shape (N, L, h, w, 2)
#     output,  # ptr to float32, shape (N, L, C, h, w)
#     N,
#     L,
#     C,
#     H,
#     W,
#     h,
#     w,
#     BLOCK_C: tl.constexpr,
# ):
#     """Takes C channels from one location (n, l, iy, ix).

#     Two program dimensions:
#         axis 0: (N*L*h*w) spatial location
#         axis 1: channel blocks of size BLOCK_C.
#     """
#     pid_spatial = tl.program_id(axis=0)  # over N * L * h * w
#     pid_cb = tl.program_id(axis=1)  # over channel blocks (C // BLOCK_C)

#     # Map `pid_spatial` → (n, l, iy, ix)
#     HW_patch = h * w
#     LW_patch = L * HW_patch

#     n = pid_spatial // LW_patch
#     rem = pid_spatial % LW_patch
#     l = rem // HW_patch
#     rem = rem % HW_patch
#     iy = rem // w  # 0 .. h-1
#     ix = rem % w  # 0 .. w-1

#     # Channel indices handled by this program
#     c_start = pid_cb * BLOCK_C
#     c_offsets = c_start + tl.arange(0, BLOCK_C)
#     mask_c = c_offsets < C

#     # Load normalized grid coordinates in the range [-1, 1]
#     # Grid linear offset: ((((n * L) + l) * h + iy) * w + ix) * 2 + {0,1}
#     grid_offset = ((((n * L) + l) * h + iy) * w + ix) * 2
#     gx = tl.load(grid + grid_offset + 0)  # x coordinate
#     gy = tl.load(grid + grid_offset + 1)  # y coordinate

#     # Convert to absolute (x, y) in image pixel space (assumes align_corners = False)
#     # x = (gx + 1) * W / 2 - 0.5
#     x_real = (gx + 1.0) * W * 0.5 - 0.5
#     y_real = (gy + 1.0) * H * 0.5 - 0.5

#     x0 = tl.floor(x_real)
#     y0 = tl.floor(y_real)
#     x1 = x0 + 1.0
#     y1 = y0 + 1.0

#     x0i = tl.clamp(x0, 0, W - 1).to(tl.int32)
#     x1i = tl.clamp(x1, 0, W - 1).to(tl.int32)
#     y0i = tl.clamp(y0, 0, H - 1).to(tl.int32)
#     y1i = tl.clamp(y1, 0, H - 1).to(tl.int32)

#     # Bilinear weights
#     wx1 = x_real - x0
#     wx0 = 1.0 - wx1
#     wy1 = y_real - y0
#     wy0 = 1.0 - wy1

#     w00 = wx0 * wy0
#     w01 = wx1 * wy0
#     w10 = wx0 * wy1
#     w11 = wx1 * wy1

#     # Gather four neighbours for every channel in the block
#     # Tensor is contiguous in (C, H, W):
#     # offset = ((((n * C) + c) * H) + y) * W + x
#     base_nc = n * C + c_offsets  # (BLOCK_C,)

#     off00 = (base_nc * H + y0i) * W + x0i
#     off01 = (base_nc * H + y0i) * W + x1i
#     off10 = (base_nc * H + y1i) * W + x0i
#     off11 = (base_nc * H + y1i) * W + x1i

#     v00 = tl.load(images + off00, mask=mask_c, other=0.0)
#     v01 = tl.load(images + off01, mask=mask_c, other=0.0)
#     v10 = tl.load(images + off10, mask=mask_c, other=0.0)
#     v11 = tl.load(images + off11, mask=mask_c, other=0.0)

#     val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11

#     # Store to output (N, L, C, h, w) contiguous
#     # offset = (((((n * L) + l) * C + c) * h + iy) * w + ix)
#     out_offset = ((((n * L) + l) * C + c_offsets) * h + iy) * w + ix
#     tl.store(output + out_offset, val, mask=mask_c)


# def grid_sample_batched_triton(
#     images: torch.Tensor,
#     grid: torch.Tensor,
#     *,
#     block_c: int = 32,
# ) -> torch.Tensor:
#     """Batched variant of torch.nn.functional.grid_sample written in Triton.

#     Assumes `align_corners=False`.

#     Args:
#         images: (N, C, H, W), float32, CUDA
#         grid: (N, L, h, w, 2), float32, CUDA, containing coordinates in [-1, 1]
#         block_c: # channels processed per Triton program (defaults to 32).

#     Returns output of shape (N, L, C, h, w).
#     """

#     assert images.is_cuda and grid.is_cuda
#     assert images.dtype == torch.float32 and grid.dtype == torch.float32
#     N, C, H, W = images.shape
#     Ng, L, h, w, two = grid.shape
#     if N != Ng or two != 2:
#         raise ValueError("Mismatched batch dimensions or last dim not equal to 2.")

#     # Ensure contiguous memory layout
#     images = images.contiguous()
#     grid = grid.contiguous()

#     output = torch.empty((N, L, C, h, w), device=images.device, dtype=images.dtype)

#     c_blocks = triton.cdiv(C, block_c)
#     launch_grid = (N * L * h * w, c_blocks)
#     _batched_grid_sample_kernel[launch_grid](
#         images,
#         grid,
#         output,
#         N,
#         L,
#         C,
#         H,
#         W,
#         h,
#         w,
#         BLOCK_C=block_c,
#         num_warps=4,
#     )

#     return output


@torch.compile
def grid_sample_batched(input_NCHW: Tensor, grid_NLhw2: Tensor) -> Tensor:
    N, C, H, W = input_NCHW.shape
    _, L, h, w, _ = grid_NLhw2.shape
    x, y = grid_NLhw2.unbind(-1)

    y = y * H / 2 + (H - 1) / 2
    x = x * W / 2 + (W - 1) / 2
    y0 = torch.floor(y)
    x0 = torch.floor(x)
    y1 = y0 + 1.0
    x1 = x0 + 1.0
    wy1 = y - y0
    wy0 = 1 - wy1
    wx1 = x - x0
    wx0 = 1 - wx1
    y0i = y0.clamp(0, H - 1).type(torch.int64)
    y1i = y1.clamp(0, H - 1).type(torch.int64)
    x0i = x0.clamp(0, W - 1).type(torch.int64)
    x1i = x1.clamp(0, W - 1).type(torch.int64)

    w00 = (wy0 * wx0).unsqueeze(1)
    w01 = (wy0 * wx1).unsqueeze(1)
    w10 = (wy1 * wx0).unsqueeze(1)
    w11 = (wy1 * wx1).unsqueeze(1)

    idx00 = y0i * W + x0i
    idx01 = y0i * W + x1i
    idx10 = y1i * W + x0i
    idx11 = y1i * W + x1i

    idx00 = einops.rearrange(idx00, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    idx01 = einops.rearrange(idx01, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    idx10 = einops.rearrange(idx10, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    idx11 = einops.rearrange(idx11, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    flat_input = input_NCHW.reshape(N, C, H * W)

    output = flat_input.gather(-1, idx00).reshape(N, C, L, h, w).mul_(w00)
    output.addcmul_(flat_input.gather(-1, idx01).reshape(N, C, L, h, w), w01)
    output.addcmul_(flat_input.gather(-1, idx10).reshape(N, C, L, h, w), w10)
    output.addcmul_(flat_input.gather(-1, idx11).reshape(N, C, L, h, w), w11)
    return output.transpose(1, 2)


def _grid_sample_batched_ref(input_NCHW: Tensor, grid_NLhw2: Tensor) -> Tensor:
    out = F.grid_sample(
        einops.repeat(input_NCHW, "N C H W -> (N L) C H W", L=grid_NLhw2.size(1)),
        grid_NLhw2.flatten(0, 1),
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return einops.rearrange(out, "(N L) C h w -> N L C h w", N=input_NCHW.size(0))


def test_grid_sample_batched():
    torch.manual_seed(0)

    iters = 100
    warmup = 10
    batch_size = 16
    channels = 3
    image_size = 128
    n_grids = 224
    grid_size = 16

    test_image = torch.randn(batch_size, channels, image_size, image_size).cuda()
    test_grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1

    torch.cuda.reset_peak_memory_stats()
    out_ref = _grid_sample_batched_ref(test_image, test_grid)
    assert out_ref.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    print(f"peak vram (ref): {torch.cuda.max_memory_allocated() // (1 << 20):,} MiB")

    torch.cuda.reset_peak_memory_stats()
    out_torch = grid_sample_batched(test_image, test_grid)
    assert out_torch.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    print(f"peak vram (torch): {torch.cuda.max_memory_allocated() // (1 << 20):,} MiB")
    print(f"error (torch): {(out_torch - out_ref).abs().max():.8f}")

    # torch.cuda.reset_peak_memory_stats()
    # out_triton = grid_sample_batched_triton(test_image, test_grid)
    # assert out_triton.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    # print(f"peak vram (triton): {torch.cuda.max_memory_allocated() // (1 << 20):,} MiB")
    # print(f"error (triton): {(out_triton - out_ref).abs().max():.8f}")

    del test_image, test_grid, out_torch, out_ref

    elapsed = 0.0
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        image = torch.randn(batch_size, channels, image_size, image_size).cuda()
        grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
        start.record()
        _ = _grid_sample_batched_ref(image, grid)
        stop.record()

        if i >= warmup:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(stop)
    print(f"time/iter (ref): {elapsed / (iters - warmup)} ms")

    elapsed = 0.0
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        image = torch.randn(batch_size, channels, image_size, image_size).cuda()
        grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
        start.record()
        _ = grid_sample_batched(image, grid)
        stop.record()

        if i >= warmup:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(stop)
    print(f"time/iter (torch): {elapsed / (iters - warmup)} ms")

    # elapsed = 0.0
    # for i in range(iters):
    #     start = torch.cuda.Event(enable_timing=True)
    #     stop = torch.cuda.Event(enable_timing=True)

    #     image = torch.randn(batch_size, channels, image_size, image_size).cuda()
    #     grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
    #     start.record()
    #     _ = grid_sample_batched_triton(image, grid)
    #     stop.record()

    #     if i >= warmup:
    #         torch.cuda.synchronize()
    #         elapsed += start.elapsed_time(stop)
    # print(f"time/iter (triton): {elapsed / (iters - warmup)} ms")


def extract_patches(
    images_NCHW: Tensor,
    y0_NL: Tensor,
    x0_NL: Tensor,
    y1_NL: Tensor,
    x1_NL: Tensor,
    h: int,
    w: int,
):
    """Extracts patches from batch of images using area-style intepolation.

    Expects coordinates (y0, x0, y1, x1) normalized to [-1.0, 1.0].

    L: number of patches per image
    h: resized patch height, in pixels
    w: resized patch width, in pixels

    Returns patches of shape (N, L, C, P, P).
    """
    assert torch.is_floating_point(images_NCHW)
    N, C, H, W = images_NCHW.shape
    _, L = y0_NL.shape
    assert y0_NL.shape == y1_NL.shape == x0_NL.shape == x1_NL.shape == (N, L)

    offsets_y = torch.linspace(0, 1, h + 1, device=images_NCHW.device)
    offsets_x = torch.linspace(0, 1, w + 1, device=images_NCHW.device)
    ys_NLh = y0_NL[..., None] + (y1_NL - y0_NL)[..., None] * offsets_y
    xs_NLw = x0_NL[..., None] + (x1_NL - x0_NL)[..., None] * offsets_x
    ys_NLhw = ys_NLh.view(N, L, h + 1, 1).expand(N, L, h + 1, w + 1)
    xs_NLhw = xs_NLw.view(N, L, 1, w + 1).expand(N, L, h + 1, w + 1)
    grid_xy_NLhw2 = torch.stack([xs_NLhw, ys_NLhw], dim=-1)

    integral_NCHW = torch.zeros((N, C, H + 1, W + 1), device=images_NCHW.device)
    integral_NCHW[:, :, 1:, 1:] = images_NCHW.cumsum(dim=-1).cumsum(dim=-2)

    integral_samples_NLChw = grid_sample_batched(
        integral_NCHW, grid_xy_NLhw2.float()
    ).reshape(N, L, C, h + 1, w + 1)

    i00 = integral_samples_NLChw[..., :-1, :-1]
    i01 = integral_samples_NLChw[..., :-1, 1:]
    i10 = integral_samples_NLChw[..., 1:, :-1]
    i11 = integral_samples_NLChw[..., 1:, 1:]

    # Box-sum via 4-corner trick, then divide by pixel area
    sums = i11 - i01 - i10 + i00
    area = (y1_NL - y0_NL) * (x1_NL - x0_NL) * (H * W / h / w / 4)
    patches = sums / area.view(N, L, 1, 1, 1)
    return patches.type_as(images_NCHW)


def main():
    from pathlib import Path

    import numpy as np
    from PIL import Image

    image_HWC = np.array(Image.open(Path(__file__).parent / "peak.jpg"))
    image_HWC = torch.tensor(image_HWC).cuda() / 255.0
    patch = extract_patches(
        einops.rearrange(image_HWC, "H W C -> 1 C H W"),
        torch.tensor(-0.9).reshape(1, 1).cuda(),
        torch.tensor(0.1).reshape(1, 1).cuda(),
        torch.tensor(-0.5).reshape(1, 1).cuda(),
        torch.tensor(0.7).reshape(1, 1).cuda(),
        h=512,
        w=512,
    )
    patch = patch.clamp(0, 1).mul(255.0).cpu().type(torch.uint8)
    patch = einops.rearrange(patch, "1 1 C h w -> h w C").numpy()
    Image.fromarray(patch).save("tmp3.png")


if __name__ == "__main__":
    test_grid_sample_batched()
