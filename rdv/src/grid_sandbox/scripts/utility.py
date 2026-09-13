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

def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def create_two_level_grid(cloud_tensor: torch.Tensor, block_size: int = 8):
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

    non_empty_mask = block_maxes > 0
    active_blocks = non_empty_mask.sum().item()

    valid_indices = non_empty_mask.nonzero().squeeze(-1)
    active_pool = flat_blocks[valid_indices].contiguous()

    # block_pool[0] is a reserved all-zero block. Empty macro cells point at it (index 0)
    # instead of at -1, so the shader can always dereference block_idx without a branch.
    empty_block = torch.zeros((1, block_size, block_size, block_size, c), dtype=cloud_tensor.dtype, device=cloud_tensor.device)
    block_pool = torch.cat([empty_block, active_pool], dim=0)

    macro_grid = torch.zeros((depth_count, height_count, width_count), dtype=torch.int32, device=cloud_tensor.device)
    macro_grid_flat = macro_grid.view(-1)

    seq_indices = torch.arange(1, active_blocks + 1, dtype=torch.int32, device=cloud_tensor.device)
    macro_grid_flat[valid_indices] = seq_indices

    macro_grid_rdv = rdv.tensor_copy(macro_grid)
    block_pool_rdv = rdv.tensor_copy(block_pool)

    return macro_grid_rdv, block_pool_rdv

# help from claude as this had to be done in numpy for runtime
def create_two_level_grid_padded(cloud_tensor: torch.Tensor, block_size: int = 8):
    spatial_dims = cloud_tensor.shape[:3]

    for dim in spatial_dims:
        if dim % block_size != 0:
            raise ValueError(f"Grid dimensions must be divisible by block_size")

    depth_count = spatial_dims[0] // block_size
    height_count = spatial_dims[1] // block_size
    width_count = spatial_dims[2] // block_size
    c = cloud_tensor.shape[-1]

    # same non-overlapping decomposition as create_two_level_grid, used only to pick active blocks
    blocks = cloud_tensor.view(depth_count, block_size, height_count, block_size, width_count, block_size, c)
    blocks = blocks.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    flat_blocks = blocks.view(-1, block_size, block_size, block_size, c)
    block_maxes = flat_blocks.abs().amax(dim=(1, 2, 3, 4))

    non_empty_mask = block_maxes > 0
    active_blocks = non_empty_mask.sum().item()
    valid_indices = non_empty_mask.nonzero().squeeze(-1)

    # zero-pad by 1 voxel on the +D/+H/+W side, then carve overlapping (block_size+1)^3 windows
    # via as_strided-backed unfold -- the padding is only ever read for a block's true last row,
    # which the shader always clamps away, so its value never actually reaches a sample.
    padded_vol = torch.nn.functional.pad(cloud_tensor.permute(3, 0, 1, 2), (0, 1, 0, 1, 0, 1))  # (c, D+1, H+1, W+1)
    windows = padded_vol.unfold(1, block_size + 1, block_size) \
                         .unfold(2, block_size + 1, block_size) \
                         .unfold(3, block_size + 1, block_size)
    # windows: (c, depth_count, height_count, width_count, block_size+1, block_size+1, block_size+1)
    windows = windows.permute(1, 2, 3, 4, 5, 6, 0)  # -> (depth_count, height_count, width_count, B+1, B+1, B+1, c)

    d_idx = valid_indices // (height_count * width_count)
    rem = valid_indices % (height_count * width_count)
    h_idx = rem // width_count
    w_idx = rem % width_count
    active_pool = windows[d_idx, h_idx, w_idx].contiguous()  # gathers only active blocks

    # block_pool[0] is a reserved all-zero (block_size+1)^3 block. Empty macro cells point at
    # it (index 0) instead of at -1, so the shader can always dereference block_idx without a branch.
    padded_size = block_size + 1
    empty_block = torch.zeros((1, padded_size, padded_size, padded_size, c), dtype=cloud_tensor.dtype, device=cloud_tensor.device)
    block_pool = torch.cat([empty_block, active_pool], dim=0)

    macro_grid = torch.zeros((depth_count, height_count, width_count), dtype=torch.int32, device=cloud_tensor.device)
    macro_grid_flat = macro_grid.view(-1)

    seq_indices = torch.arange(1, active_blocks + 1, dtype=torch.int32, device=cloud_tensor.device)
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