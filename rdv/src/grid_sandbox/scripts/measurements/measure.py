"""
Usage:
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

STEP_SIZE = 0.001
IMAGE_SIZE = 512
NUM_POINTS = 256 ** 3
QUERY_BOX_SHAPE = (256, 256, 256)  # bounding box for random/linear point queries, independent of the volume's own shape
MAX_BLOCK_SIZE = 32  # .nvdb files are cropped to a multiple of 32 (see nvdb_converter.py)

VARIANTS = ["dense", "two_level", "two_level_padded", "nanovdb", "nanovdb_onefetch", "nanovdb_onefetch_notrilinear",
            "two_level_dda", "two_level_dda_padded", "nanovdb_dda",
            "null"]  # null is a point-query diagnostic (the I/O floor), see build_grid
QUERY_VARIANTS = ["dense", "two_level", "two_level_padded", "nanovdb", "nanovdb_onefetch",
                  "nanovdb_onefetch_notrilinear", "null"]
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
    if variant == "dense":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        grid = imp.DenseGrid3D(rdv.tensor_copy(vol))

    elif variant == "two_level":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        macro_grid, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=block_size)
        grid = imp.TwoLevelGrid3D(macro_grid=macro_grid, block_pool=block_pool, block_size=block_size)

    elif variant == "two_level_padded":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        macro_grid, block_pool = create_two_level_grid_padded(rdv.tensor_copy(vol), block_size=block_size)
        grid = imp.TwoLevelGrid3DPadded(macro_grid=macro_grid, block_pool=block_pool, block_size=block_size)

    elif variant == "nanovdb":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
        nvdb_bytes = np.fromfile(nvdb_path, dtype=np.uint8)
        nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes))
        D, H, W, C = vol.shape
        grid = imp.NanoVDBGrid3D(nvdb_tensor, shape=(D, H, W), align_corners=True)

    elif variant == "nanovdb_onefetch":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
        nvdb_bytes = np.fromfile(nvdb_path, dtype=np.uint8)
        nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes))
        D, H, W, C = vol.shape
        grid = imp.NanoVDBGrid3DOneFetch(nvdb_tensor, shape=(D, H, W), align_corners=True)

    elif variant == "nanovdb_onefetch_notrilinear":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        nvdb_path = os.path.join(DATA_DIR, f"{volume_name}.nvdb")
        nvdb_bytes = np.fromfile(nvdb_path, dtype=np.uint8)
        nvdb_tensor = rdv.tensor_copy(torch.from_numpy(nvdb_bytes))
        D, H, W, C = vol.shape
        grid = imp.NanoVDBGrid3DOneFetchNoTrilinear(nvdb_tensor, shape=(D, H, W), align_corners=True)

    elif variant == "null":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        grid = imp.NullSampler3D(rdv.tensor_copy(vol))

    else:
        raise ValueError(f"--mode random/linear only supports --variant in {QUERY_VARIANTS}, got {variant!r}")

    return grid, tuple(vol.shape)


def build_view(variant, volume_name, block_size):
    if variant in QUERY_VARIANTS:
        grid, shape = build_grid(variant, volume_name, block_size)
        D, H, W, C = shape
        transform = build_transform((D, H, W))
        map_ = rdv.RaymarchingTransmittance(extinction=grid * 5.0, step_size=STEP_SIZE, transform=transform)

    elif variant == "two_level_dda":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        D, H, W, C = vol.shape
        transform = build_transform((D, H, W))
        macro_grid, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=block_size)
        map_ = imp.RaymarchingTransmittanceTwoLevelDDA(
            macro_grid, block_pool, block_size, step_size=STEP_SIZE, transform=transform, extinction_scale=5.0
        )

    elif variant == "two_level_dda_padded":
        vol = load_trimmed_volume(volume_name, block_size=MAX_BLOCK_SIZE)
        D, H, W, C = vol.shape
        transform = build_transform((D, H, W))
        macro_grid, block_pool = create_two_level_grid_padded(rdv.tensor_copy(vol), block_size=block_size)
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


def measure(variant, volume, block_size, mode="render", warmup=5, iters=25):
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

        # overhead fix ?
        points_wrapped = vk.wrap_gpu(points, 'in')

        call = lambda: grid(points)
        num_points = points.shape[0]

    D, H, W, C = shape
    vol = load_trimmed_volume(volume, MAX_BLOCK_SIZE)
    active_voxels = (vol.abs() > 0).sum().item()
    total_voxels = vol.numel()

    active_blocks = total_blocks = None
    if variant in ("two_level", "two_level_dda", "two_level_padded", "two_level_dda_padded"):
        _, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=block_size)
        active_blocks = block_pool.shape[0] - 1  # block_pool[0] is the reserved empty block
        total_blocks = (D // block_size) * (H // block_size) * (W // block_size)

    # this is the actual measuring functionality
    with torch.no_grad():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()

        times_ms = []
        for i in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

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
    parser.add_argument("--volume", default="cloud_760")
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
