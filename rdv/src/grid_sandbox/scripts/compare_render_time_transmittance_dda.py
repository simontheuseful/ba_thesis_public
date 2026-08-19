
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

BLOCK_SIZES = [8, 16, 32]
TRIM_BLOCK_SIZE = max(BLOCK_SIZES)  # volume dims are trimmed to a multiple of this
THRESHOLD = 1e-4
STEP_SIZE = 0.001
IMAGE_SIZE = 512
WARMUP_ITERS = 5
MEASURE_ITERS = 10
EXTINCTION_SCALE = 5.0

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
    D, H, W = (D // TRIM_BLOCK_SIZE) * TRIM_BLOCK_SIZE, (H // TRIM_BLOCK_SIZE) * TRIM_BLOCK_SIZE, (W // TRIM_BLOCK_SIZE) * TRIM_BLOCK_SIZE
    vol = vol[:D, :H, :W, :]

    active_voxels = (vol.abs() > THRESHOLD).sum().item()
    total_voxels = vol.numel()

    transform = build_transform((D, H, W))
    sensor = build_sensor()

    vol_rdv = rdv.tensor_copy(vol)

    results = {}
    block_occupancy = {}

    for bs in BLOCK_SIZES:
        macro_grid, block_pool = create_two_level_grid(vol_rdv, block_size=bs, threshold=THRESHOLD)
        two_level_bytes = tensor_bytes(macro_grid) + tensor_bytes(block_pool)
        block_occupancy[bs] = (block_pool.shape[0], (D // bs) * (H // bs) * (W // bs))

        sparse_grid = imp.TwoLevelGrid3D(macro_grid=macro_grid, block_pool=block_pool, block_size=bs)
        sparse_map = rdv.RaymarchingTransmittance(extinction=sparse_grid * EXTINCTION_SCALE, step_size=STEP_SIZE, transform=transform)
        results[f"2-L (bs={bs})"] = (*time_render_ms(sensor.view(sparse_map)), two_level_bytes)

        sparse_dda_map = imp.RaymarchingTransmittanceTwoLevelDDA(
            macro_grid, block_pool, bs, step_size=STEP_SIZE, transform=transform, extinction_scale=EXTINCTION_SCALE)
        results[f"2-L (bs={bs}, DDA)"] = (*time_render_ms(sensor.view(sparse_dda_map)), two_level_bytes)

    nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")

    nvdb_bytes_raw = np.fromfile(nvdb_path, dtype=np.uint8)
    nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes_raw))

    nvdb_grid = imp.NanoVDBGrid3D(nvdb_tensor, shape=(D, H, W), align_corners=True)
    nvdb_map = rdv.RaymarchingTransmittance(extinction=nvdb_grid * EXTINCTION_SCALE, step_size=STEP_SIZE, transform=transform)
    nvdb_bytes = tensor_bytes(nvdb_grid.nvdb_data)
    results["NanoVDB"] = (*time_render_ms(sensor.view(nvdb_map)), nvdb_bytes)

    nvdb_dda_map = imp.RaymarchingTransmittanceNanoVDBDDA(
        nvdb_tensor, shape=(D, H, W), step_size=STEP_SIZE, transform=transform, extinction_scale=EXTINCTION_SCALE)
    results["NanoVDB (DDA)"] = (*time_render_ms(sensor.view(nvdb_dda_map)), nvdb_bytes)


    print(f"Volume: {volume_name}   shape: {(D, H, W, C)}   sparsity: {active_voxels}/{total_voxels} voxels "
          f"above threshold ({active_voxels / total_voxels * 100:.1f}%)")
    print(f"{'':<20}{'render (ms/frame, mean +/- std)':>32}{'size (MB)':>14}")
    for label, (ms, std, _, size_bytes) in results.items():
        print(f"{label:<20}{ms:>20.3f} +/- {std:<8.3f}{size_bytes / 2**20:>14.2f}")

    print()
    for bs in BLOCK_SIZES:
        active_blocks, total_blocks = block_occupancy[bs]
        print(f"2-L (bs={bs}) occupancy: {active_blocks}/{total_blocks} blocks "
              f"({active_blocks / total_blocks * 100:.1f}%)")

    print()
    for base_label, dda_label in [(f"2-L (bs={bs})", f"2-L (bs={bs}, DDA)") for bs in BLOCK_SIZES] + [("NanoVDB", "NanoVDB (DDA)")]:
        if base_label in results and dda_label in results:
            speedup = results[base_label][0] / results[dda_label][0]
            print(f"{base_label:<16} -> DDA speedup: {speedup:.2f}x")

    labels = list(results.keys())
    render_time = [v[0] for v in results.values()]
    render_time_std = [v[1] for v in results.values()]
    size_mb = [v[3] / 2**20 for v in results.values()]

    plot_comparison(labels, render_time, size_mb, volume_name, DIAGRAMS_DIR, shape=(D, H, W),
                                    render_time_std=render_time_std,
                                    title="Grid representation comparison (DDA empty-space skipping)",
                                    file_suffix="_dda", sparsity=active_voxels / total_voxels)


if __name__ == "__main__":
    main()
