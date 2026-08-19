
import os
import sys
import numpy as np
import torch
import rdv
import grid_implementations as imp
from utility import create_two_level_grid, load_pt_volume, time_calls_ms
from plotting import plot_comparison

_HERE = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS_DIR = os.path.join(_HERE, "..", "diagrams")
DATA_DIR = os.path.join(_HERE, "..", "data")

TWO_LEVEL_BLOCK_SIZES = [1, 4, 8, 16, 32]
BLOCK_SIZE = max(TWO_LEVEL_BLOCK_SIZES)  # volume dims are trimmed to a multiple of this
THRESHOLD = 1e-4
NUM_POINTS = 256 ** 3
WARMUP_ITERS = 5   # discarded, not recorded
MEASURE_ITERS = 10  # recorded individually, to get mean + std

DEFAULT_VOLUME = "cloud_865"


def tensor_bytes(t):
    return t.numel() * t.element_size()


def run_query_comparison(point_generator, title, file_suffix, volume_name=None, num_points=NUM_POINTS):
    volume_name = volume_name or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOLUME)
    pt_path = os.path.join(DATA_DIR, f"{volume_name}.pt")

    vol = load_pt_volume(pt_path)
    D, H, W, C = vol.shape
    D, H, W = (D // BLOCK_SIZE) * BLOCK_SIZE, (H // BLOCK_SIZE) * BLOCK_SIZE, (W // BLOCK_SIZE) * BLOCK_SIZE
    vol = vol[:D, :H, :W, :]

    # voxel-level sparsity: structure-independent, this is what any of the sparse
    # representations below actually have to work with regardless of block_size.
    active_voxels = (vol.abs() > THRESHOLD).sum().item()
    total_voxels = vol.numel()

    points = point_generator(D, H, W, num_points)
    num_points = points.shape[0]

    vol_dense_rdv = rdv.tensor_copy(vol)
    dense_grid = imp.ExperimentalGrid3D(vol_dense_rdv)
    dense_query_ms, dense_query_std, _ = time_calls_ms(
        lambda: dense_grid(points), warmup=WARMUP_ITERS, iterations=MEASURE_ITERS)
    dense_bytes = tensor_bytes(vol_dense_rdv)

    two_level_results = []
    for bs in TWO_LEVEL_BLOCK_SIZES:
        vol_sparse_rdv = rdv.tensor_copy(vol)
        macro_grid, block_pool = create_two_level_grid(vol_sparse_rdv, block_size=bs, threshold=THRESHOLD)
        sparse_grid = imp.TwoLevelGrid3D(macro_grid=macro_grid, block_pool=block_pool, block_size=bs)

        sparse_query_ms, sparse_query_std, _ = time_calls_ms(
            lambda: sparse_grid(points), warmup=WARMUP_ITERS, iterations=MEASURE_ITERS)

        active_blocks = block_pool.shape[0]
        total_blocks = (D // bs) * (H // bs) * (W // bs)
        sparse_bytes = tensor_bytes(macro_grid) + tensor_bytes(block_pool)

        two_level_results.append(dict(
            block_size=bs, query_ms=sparse_query_ms, query_std=sparse_query_std,
            active_blocks=active_blocks, total_blocks=total_blocks, size_bytes=sparse_bytes,
        ))

    labels = ["Dense"] + [f"2-L (bs={r['block_size']})" for r in two_level_results]
    query_time = [dense_query_ms] + [r['query_ms'] for r in two_level_results]
    query_time_std = [dense_query_std] + [r['query_std'] for r in two_level_results]
    size_mb = [dense_bytes / 2**20] + [r['size_bytes'] / 2**20 for r in two_level_results]

    nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
    nvdb_bytes_raw = np.fromfile(nvdb_path, dtype=np.uint8)
    nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes_raw))
    nvdb_grid = imp.NanoVDBGrid3D(nvdb_tensor, shape=(D, H, W), align_corners=True)
    nvdb_query_ms, nvdb_query_std, _ = time_calls_ms(
        lambda: nvdb_grid(points), warmup=WARMUP_ITERS, iterations=MEASURE_ITERS)
    nvdb_size_bytes = tensor_bytes(nvdb_grid.nvdb_data)
    labels.append("NanoVDB")
    query_time.append(nvdb_query_ms)
    query_time_std.append(nvdb_query_std)
    size_mb.append(nvdb_size_bytes / 2**20)

    print(f"Volume: {volume_name}   shape: {(D, H, W, C)}   points: {num_points}   sparsity: "
          f"{active_voxels}/{total_voxels} voxels above threshold ({active_voxels / total_voxels * 100:.1f}%)")
    print(f"{'':<16}{'query (ms/batch, mean +/- std)':>32}")
    print(f"{'Dense grid':<16}{dense_query_ms:>20.3f} +/- {dense_query_std:.3f}")
    for r in two_level_results:
        label = f"2-level (bs={r['block_size']})"
        print(f"{label:<16}{r['query_ms']:>20.3f} +/- {r['query_std']:.3f}")
    print(f"{'NanoVDB grid':<16}{nvdb_query_ms:>20.3f} +/- {nvdb_query_std:.3f}")

    print(f"\n{'':<16}{'size (MB)':>12}{'vs dense':>12}  active tiles")
    print(f"{'Dense grid':<16}{dense_bytes / 2**20:>12.2f}{'1.00x':>12}")
    for r in two_level_results:
        label = f"2-level (bs={r['block_size']})"
        ratio = f"{dense_bytes / r['size_bytes']:.2f}x"
        active = f"{r['active_blocks']}/{r['total_blocks']} ({r['active_blocks'] / r['total_blocks'] * 100:.1f}%)"
        print(f"{label:<16}{r['size_bytes'] / 2**20:>12.2f}{ratio:>12}  {active}")
    print(f"{'NanoVDB grid':<16}{nvdb_size_bytes / 2**20:>12.2f}{f'{dense_bytes / nvdb_size_bytes:.2f}x':>12}")

    plot_comparison(labels, query_time, size_mb, volume_name, DIAGRAMS_DIR, shape=(D, H, W),
                                    render_time_std=query_time_std, title=title,
                                    time_label="Query Time per Batch", file_suffix=file_suffix,
                                    sparsity=active_voxels / total_voxels)
