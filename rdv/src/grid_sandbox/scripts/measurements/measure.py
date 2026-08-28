"""
Main measurement script: times one grid implementation, one configuration, console output only.

--mode render (default) times sensor.view(map).capture(), i.e. a full camera raymarch,
for any of VARIANTS.

--mode random / linear time grid(points) directly -- a batch of point queries against
the grid, no camera/raymarching involved -- for --variant in {dense, two_level, nanovdb}
only (two_level_dda / nanovdb_dda take a 6D ray, not a 3D point, so they have no
point-query form).

Usage (from this directory, with rdv/src on PYTHONPATH):
    python measure.py --variant dense
    python measure.py --variant two_level --block-size 8
    python measure.py --variant nanovdb_dda --volume disney_cloud
    python measure.py --variant two_level --block-size 8 --mode random
    python measure.py --variant nanovdb --mode linear
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # grid_implementations.py / utility.py live one level up

import numpy as np
import torch
import rdv
import vulky as vk
import grid_implementations as imp
from utility import (create_two_level_grid, create_two_level_grid_padded, load_pt_volume,
                      generate_random_points, generate_linear_points)

DATA_DIR = os.path.join(_HERE, "..", "..", "data")

THRESHOLD = 1e-4
STEP_SIZE = 0.001
IMAGE_SIZE = 512
NUM_POINTS = 256 ** 3
QUERY_BOX_SHAPE = (256, 256, 256)  # bounding box for random/linear point queries, independent of the volume's own shape
MAX_BLOCK_SIZE = 32  # .nvdb files are cropped to a multiple of 32 (see nvdb_converter.py)

VARIANTS = ["dense", "two_level", "two_level_padded", "nanovdb", "two_level_dda", "two_level_dda_padded", "nanovdb_dda"]
QUERY_VARIANTS = ["dense", "two_level", "two_level_padded", "nanovdb"]
MODES = ["render", "random", "linear"]


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


def load_trimmed_volume(volume_name, block_size):
    pt_path = os.path.join(DATA_DIR, f"{volume_name}.pt")
    vol = load_pt_volume(pt_path)
    D, H, W, C = vol.shape
    D, H, W = (D // block_size) * block_size, (H // block_size) * block_size, (W // block_size) * block_size
    return vol[:D, :H, :W, :]


def build_grid(variant, volume_name, block_size):
    """Builds just the grid map (no raymarching) for the point-query variants."""
    if variant == "dense":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        grid = imp.ExperimentalGrid3D(rdv.tensor_copy(vol))

    elif variant == "two_level":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        macro_grid, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=block_size, threshold=THRESHOLD)
        grid = imp.TwoLevelGrid3D(macro_grid=macro_grid, block_pool=block_pool, block_size=block_size)

    elif variant == "two_level_padded":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        macro_grid, block_pool = create_two_level_grid_padded(rdv.tensor_copy(vol), block_size=block_size, threshold=THRESHOLD)
        grid = imp.TwoLevelGrid3DPadded(macro_grid=macro_grid, block_pool=block_pool, block_size=block_size)

    elif variant == "nanovdb":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
        nvdb_bytes = np.fromfile(nvdb_path, dtype=np.uint8)
        nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes))
        D, H, W, C = vol.shape
        grid = imp.NanoVDBGrid3D(nvdb_tensor, shape=(D, H, W), align_corners=True)

    else:
        raise ValueError(f"--mode random/linear only supports --variant in {QUERY_VARIANTS}, got {variant!r}")

    return grid, tuple(vol.shape)


def build_view(variant, volume_name, block_size):
    if variant in QUERY_VARIANTS:
        grid, shape = build_grid(variant, volume_name, block_size)
        D, H, W, C = shape
        transform = build_transform((D, H, W))
        map_ = rdv.RaymarchingTransmittance(extinction=grid * 5, step_size=STEP_SIZE, transform=transform)

    elif variant == "two_level_dda":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        D, H, W, C = vol.shape
        transform = build_transform((D, H, W))
        macro_grid, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=block_size, threshold=THRESHOLD)
        map_ = imp.RaymarchingTransmittanceTwoLevelDDA(
            macro_grid, block_pool, block_size, step_size=STEP_SIZE, transform=transform, extinction_scale=5.0
        )

    elif variant == "two_level_dda_padded":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        D, H, W, C = vol.shape
        transform = build_transform((D, H, W))
        macro_grid, block_pool = create_two_level_grid_padded(rdv.tensor_copy(vol), block_size=block_size, threshold=THRESHOLD)
        map_ = imp.RaymarchingTransmittanceTwoLevelDDAPadded(
            macro_grid, block_pool, block_size, step_size=STEP_SIZE, transform=transform, extinction_scale=5.0
        )

    elif variant == "nanovdb_dda":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        D, H, W, C = vol.shape
        transform = build_transform((D, H, W))
        nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
        nvdb_bytes = np.fromfile(nvdb_path, dtype=np.uint8)
        nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes))
        map_ = imp.RaymarchingTransmittanceNanoVDBDDA(
            nvdb_tensor, shape=(D, H, W), step_size=STEP_SIZE, transform=transform,
            align_corners=True, extinction_scale=5.0
        )

    else:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")

    return build_sensor().view(map_), (D, H, W, C)


def measure(variant, volume, block_size, mode="render", warmup=3, iters=3):
    """Builds the given configuration, times it, and returns a result dict. Raises
    ValueError for an invalid block_size/mode/variant combination."""
    if not (1 <= block_size <= MAX_BLOCK_SIZE):
        raise ValueError(f"block_size must be between 1 and {MAX_BLOCK_SIZE}, got {block_size}")

    if mode == "render":
        view, shape = build_view(variant, volume, block_size)
        call = view.capture
        num_points = None
    else:
        if variant not in QUERY_VARIANTS:
            raise ValueError(f"mode {mode!r} only supports variant in {QUERY_VARIANTS}, got {variant!r}")
        grid, shape = build_grid(variant, volume, block_size)
        D, H, W, C = shape
        points = generate_random_points(NUM_POINTS) if mode == "random" else generate_linear_points(QUERY_BOX_SHAPE)
        # Pre-wrap points once and hold the wrapper alive for the rest of this call: vulky's
        # wrap_gpu() caches by points.data_ptr() in a weakref.WeakSet, but nothing else keeps
        # the wrapper alive between grid(points) calls, so without this the cache misses every
        # time and the full points tensor gets needlessly re-copied to GPU on every call --
        # including every measured iteration, not just once during warmup.
        points_wrapped = vk.wrap_gpu(points, 'in')
        call = lambda: grid(points)
        num_points = points.shape[0]

    D, H, W, C = shape
    vol = load_trimmed_volume(volume, MAX_BLOCK_SIZE)
    active_voxels = (vol.abs() > THRESHOLD).sum().item()
    total_voxels = vol.numel()

    active_blocks = total_blocks = None
    if variant in ("two_level", "two_level_dda", "two_level_padded", "two_level_dda_padded"):
        # active_blocks/total_blocks only depend on which blocks are active, not on padding,
        # so the plain (unpadded) grid builder is enough here even for two_level_padded.
        _, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=block_size, threshold=THRESHOLD)
        active_blocks = block_pool.shape[0]
        total_blocks = (D // block_size) * (H // block_size) * (W // block_size)

    with torch.no_grad():
        torch.cuda.nvtx.range_push("warmup")
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        times_ms = []
        for i in range(iters):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"measured_{i}")
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            torch.cuda.nvtx.range_pop()

    times_ms = np.array(times_ms)
    return dict(
        variant=variant, mode=mode, volume=volume, shape=shape, block_size=block_size,
        num_points=num_points, active_voxels=active_voxels, total_voxels=total_voxels,
        active_blocks=active_blocks, total_blocks=total_blocks,
        mean_ms=float(times_ms.mean()), std_ms=float(times_ms.std()), n=len(times_ms),
    )


def format_result(r):
    extra = f" points={r['num_points']}" if r['num_points'] is not None else ""
    stats = (f" active_voxels={r['active_voxels']}/{r['total_voxels']} "
             f"({r['active_voxels'] / r['total_voxels'] * 100:.1f}%)")
    if r['active_blocks'] is not None:
        stats += (f" active_blocks={r['active_blocks']}/{r['total_blocks']} "
                  f"({r['active_blocks'] / r['total_blocks'] * 100:.1f}%)")

    line1 = (f"variant={r['variant']} mode={r['mode']} volume={r['volume']} shape={r['shape']} "
             f"block_size={r['block_size']}{extra}{stats}")
    line2 = f"{r['mode']}: {r['mean_ms']:.3f} +/- {r['std_ms']:.3f} ms  (n={r['n']})"
    return f"{line1}\n{line2}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mode", choices=MODES, default="render")
    parser.add_argument("--volume", default="cloud_356")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    try:
        result = measure(args.variant, args.volume, args.block_size, args.mode, args.warmup, args.iters)
    except ValueError as e:
        parser.error(str(e))
        return

    print(format_result(result))


if __name__ == "__main__":
    main()
