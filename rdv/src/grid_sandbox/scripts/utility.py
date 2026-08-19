import gc
import time
import numpy as np
import torch
import rdv
import warp as wp

GRID_MAGIC = (0x304244566f6e614e, 0x314244566f6e614e)  # "NanoVDB0" / "NanoVDB1" -- bare GridData


def strip_nvdb_header(nvdb_data: torch.Tensor) -> torch.Tensor:
    cpu_bytes = nvdb_data.cpu()
    magic = int.from_bytes(bytes(cpu_bytes[:8].numpy()), "little")
    if magic in GRID_MAGIC:
        grid_data = cpu_bytes  # already bare GridData, nothing to strip
    else:
        wp.init()
        volume = wp.Volume.load_from_nvdb(bytes(cpu_bytes.numpy()), device="cpu")
        grid_data = torch.from_numpy(volume.array().numpy())
    return rdv.tensor_copy(grid_data.contiguous())


def load_pt_volume(path: str, device="cuda") -> torch.Tensor:
    return torch.load(path, map_location=device, weights_only=True)


def create_two_level_grid(cloud_tensor: torch.Tensor, block_size: int = 8, threshold: float = 1e-4):
    spatial_dims = cloud_tensor.shape[:3]

    for dim in spatial_dims:
        if dim % block_size != 0:
            raise ValueError(f"Grid dimensions must be divisible by block_size")

    depth_count = spatial_dims[0] // block_size
    height_count = spatial_dims[1] // block_size
    width_count = spatial_dims[2] // block_size
    c = cloud_tensor.shape[-1]

    # rearrange the tensor to group blocks together
    blocks = cloud_tensor.view(depth_count, block_size, height_count, block_size, width_count, block_size, c)
    blocks = blocks.permute(0, 2, 4, 1, 3, 5, 6).contiguous()

    flat_blocks = blocks.view(-1, block_size, block_size, block_size, c)

    block_maxes = flat_blocks.abs().amax(dim=(1, 2, 3, 4))

    non_empty_mask = block_maxes > threshold
    active_blocks = non_empty_mask.sum().item()

    valid_indices = non_empty_mask.nonzero().squeeze(-1)
    block_pool = flat_blocks[valid_indices].contiguous()

    macro_grid = torch.full((depth_count, height_count, width_count), -1, dtype=torch.int32, device=cloud_tensor.device)
    macro_grid_flat = macro_grid.view(-1)

    seq_indices = torch.arange(active_blocks, dtype=torch.int32, device=cloud_tensor.device)
    macro_grid_flat[valid_indices] = seq_indices

    macro_grid_rdv = rdv.tensor_copy(macro_grid)
    block_pool_rdv = rdv.tensor_copy(block_pool)

    return macro_grid_rdv, block_pool_rdv


def generate_random_points(n: int, device="cuda", seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(n, 3, generator=g, device=device) * 2 - 1


def generate_linear_points(shape, device="cuda") -> torch.Tensor:
    D, H, W = shape
    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).contiguous()


def time_calls_ms(fn, warmup: int = 5, iterations: int = 10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        per_call_ms = np.empty(iterations)
        for i in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            per_call_ms[i] = (time.perf_counter() - t0) * 1000.0

    return float(per_call_ms.mean()), float(per_call_ms.std()), per_call_ms