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
STEP_SIZE = 0.001
IMAGE_SIZE = 512
WARMUP_ITERS = 5   # discarded, not recorded
MEASURE_ITERS = 10  # recorded individually, to get mean + std

DEFAULT_VOLUME = "cloud_865"


def build_sensor():
    camera_poses = rdv.tensor_from(
        [[0.2, 0.3, -2.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    )
    return rdv.Sensor(
        1, IMAGE_SIZE, IMAGE_SIZE,
        samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.CENTER, rdv.SampleLocation.CENTER),
        probes_map=rdv.CameraProbes(camera_poses=camera_poses)
    )


def build_transform(shape):
    bmin, bmax = rdv.grid3d_fit_box(shape)
    scale, offset = rdv.unit2box(bmin, bmax)
    return rdv.mat4x3.trs(offset=offset, scale=scale)


def time_render_ms(view, warmup=WARMUP_ITERS, iterations=MEASURE_ITERS):
    return time_calls_ms(view.capture, warmup=warmup, iterations=iterations)


def tensor_bytes(t):
    return t.numel() * t.element_size()


def main(volume_name=None):
    volume_name = volume_name or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOLUME)
    pt_path = os.path.join(DATA_DIR, f"{volume_name}.pt")

    vol = load_pt_volume(pt_path)
    D, H, W, C = vol.shape
    D, H, W = (D // BLOCK_SIZE) * BLOCK_SIZE, (H // BLOCK_SIZE) * BLOCK_SIZE, (W // BLOCK_SIZE) * BLOCK_SIZE
    vol = vol[:D, :H, :W, :]

    active_voxels = (vol.abs() > THRESHOLD).sum().item()
    total_voxels = vol.numel()

    transform = build_transform((D, H, W))
    sensor = build_sensor()

    vol_dense_rdv = rdv.tensor_copy(vol)
    dense_grid = imp.ExperimentalGrid3D(vol_dense_rdv)
    dense_map = rdv.RaymarchingTransmittance(extinction=dense_grid * 5, step_size=STEP_SIZE, transform=transform)
    dense_view = sensor.view(dense_map)
    dense_render_ms, dense_render_std, _ = time_render_ms(dense_view)
    dense_bytes = tensor_bytes(vol_dense_rdv)

    two_level_results = []
    for bs in TWO_LEVEL_BLOCK_SIZES:
        vol_sparse_rdv = rdv.tensor_copy(vol)
        macro_grid, block_pool = create_two_level_grid(vol_sparse_rdv, block_size=bs, threshold=THRESHOLD)
        sparse_grid = imp.TwoLevelGrid3D(macro_grid=macro_grid, block_pool=block_pool, block_size=bs)

        sparse_map = rdv.RaymarchingTransmittance(extinction=sparse_grid * 5, step_size=STEP_SIZE, transform=transform)
        sparse_view = sensor.view(sparse_map)
        sparse_render_ms, sparse_render_std, _ = time_render_ms(sparse_view)

        active_blocks = block_pool.shape[0]
        total_blocks = (D // bs) * (H // bs) * (W // bs)
        sparse_bytes = tensor_bytes(macro_grid) + tensor_bytes(block_pool)

        two_level_results.append(dict(
            block_size=bs, render_ms=sparse_render_ms, render_std=sparse_render_std,
            active_blocks=active_blocks, total_blocks=total_blocks, size_bytes=sparse_bytes,
        ))

    labels = ["Dense"] + [f"2-L (bs={r['block_size']})" for r in two_level_results]
    render_time = [dense_render_ms] + [r['render_ms'] for r in two_level_results]
    render_time_std = [dense_render_std] + [r['render_std'] for r in two_level_results]
    size_mb = [dense_bytes / 2**20] + [r['size_bytes'] / 2**20 for r in two_level_results]

    nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
    nvdb_bytes_raw = np.fromfile(nvdb_path, dtype=np.uint8)
    nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes_raw))
    nvdb_grid = imp.NanoVDBGrid3D(nvdb_tensor, shape=(D, H, W), align_corners=True)
    nvdb_map = rdv.RaymarchingTransmittance(extinction=nvdb_grid * 5, step_size=STEP_SIZE, transform=transform)
    nvdb_view = sensor.view(nvdb_map)
    nvdb_render_ms, nvdb_render_std, _ = time_render_ms(nvdb_view)
    nvdb_size_bytes = tensor_bytes(nvdb_grid.nvdb_data)
    labels.append("NanoVDB")
    render_time.append(nvdb_render_ms)
    render_time_std.append(nvdb_render_std)
    size_mb.append(nvdb_size_bytes / 2**20)

    print(f"Volume: {volume_name}   shape: {(D, H, W, C)}   sparsity: {active_voxels}/{total_voxels} voxels "
          f"above threshold ({active_voxels / total_voxels * 100:.1f}%)")
    print(f"{'':<16}{'render (ms/frame, mean +/- std)':>32}")
    print(f"{'Dense grid':<16}{dense_render_ms:>20.3f} +/- {dense_render_std:.3f}")
    for r in two_level_results:
        label = f"2-level (bs={r['block_size']})"
        print(f"{label:<16}{r['render_ms']:>20.3f} +/- {r['render_std']:.3f}")
    print(f"{'NanoVDB grid':<16}{nvdb_render_ms:>20.3f} +/- {nvdb_render_std:.3f}")

    print(f"\n{'':<16}{'size (MB)':>12}{'vs dense':>12}  active tiles")
    print(f"{'Dense grid':<16}{dense_bytes / 2**20:>12.2f}{'1.00x':>12}")
    for r in two_level_results:
        label = f"2-level (bs={r['block_size']})"
        ratio = f"{dense_bytes / r['size_bytes']:.2f}x"
        active = f"{r['active_blocks']}/{r['total_blocks']} ({r['active_blocks'] / r['total_blocks'] * 100:.1f}%)"
        print(f"{label:<16}{r['size_bytes'] / 2**20:>12.2f}{ratio:>12}  {active}")
    print(f"{'NanoVDB grid':<16}{nvdb_size_bytes / 2**20:>12.2f}{f'{dense_bytes / nvdb_size_bytes:.2f}x':>12}")

    plot_comparison(labels, render_time, size_mb, volume_name, DIAGRAMS_DIR, shape=(D, H, W),
                                    render_time_std=render_time_std, sparsity=active_voxels / total_voxels)


if __name__ == "__main__":
    main()
