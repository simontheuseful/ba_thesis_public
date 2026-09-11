"""
Usage:
    python compare_sizes.py --volume cloud_865
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # utility.py lives one level up

import rdv
from utility import load_pt_volume, create_two_level_grid, create_two_level_grid_padded, tensor_bytes

DATA_DIR = os.path.join(_HERE, "..", "..", "data")

BLOCK_SIZE = 32  # .nvdb files are cropped to a multiple of 32 (see nvdb_converter.py)
TWO_LEVEL_BLOCK_SIZES = [1, 2, 4, 8, 16, 32]


def mib(n_bytes):
    return n_bytes / 2 ** 20


def kib(n_bytes):
    return n_bytes / 2 ** 10


def compute_sizes(volume):
    pt_path = os.path.join(DATA_DIR, f"{volume}.pt")
    nvdb_path = os.path.join(DATA_DIR, f"{volume}.nvdb")

    vol = load_pt_volume(pt_path)
    D, H, W, C = vol.shape
    D = (D // BLOCK_SIZE) * BLOCK_SIZE
    H = (H // BLOCK_SIZE) * BLOCK_SIZE
    W = (W // BLOCK_SIZE) * BLOCK_SIZE
    vol = vol[:D, :H, :W, :]

    pt_bytes = tensor_bytes(vol)
    nvdb_bytes = os.path.getsize(nvdb_path)

    active_voxels = (vol.abs() > 0).sum().item()
    total_voxels = vol.numel()

    two_level = []
    two_level_padded = []
    for bs in TWO_LEVEL_BLOCK_SIZES:
        macro_grid, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=bs)
        macro_bytes = tensor_bytes(macro_grid)
        micro_bytes = tensor_bytes(block_pool)
        active_blocks = block_pool.shape[0] - 1  # block_pool[0] is the reserved empty block
        total_blocks = (D // bs) * (H // bs) * (W // bs)
        two_level.append(dict(
            block_size=bs, macro_bytes=macro_bytes, micro_bytes=micro_bytes,
            active_blocks=active_blocks, total_blocks=total_blocks,
        ))

        macro_grid_p, block_pool_p = create_two_level_grid_padded(rdv.tensor_copy(vol), block_size=bs)
        two_level_padded.append(dict(
            block_size=bs, macro_bytes=tensor_bytes(macro_grid_p), micro_bytes=tensor_bytes(block_pool_p),
            active_blocks=block_pool_p.shape[0] - 1, total_blocks=total_blocks,
        ))

    return dict(
        volume=volume, shape=(D, H, W, C), pt_bytes=pt_bytes, nvdb_bytes=nvdb_bytes,
        active_voxels=active_voxels, total_voxels=total_voxels,
        two_level=two_level, two_level_padded=two_level_padded,
    )


def print_sizes(data):
    print(f"volume={data['volume']} shape={data['shape']}")
    print(f"active_voxels={data['active_voxels']}/{data['total_voxels']} "
          f"({data['active_voxels'] / data['total_voxels'] * 100:.1f}%)")
    print(f"{'dense (.pt)':<20}{mib(data['pt_bytes']):>10.2f} MiB")
    print(f"{'.nvdb':<20}{mib(data['nvdb_bytes']):>10.2f} MiB  "
          f"({data['pt_bytes'] / data['nvdb_bytes']:.2f}x vs dense)")

    print(f"\n{'block_size':<12}{'active_blocks':<28}{'macro (KiB)':>12}{'micro (MiB)':>12}"
          f"{'total (MiB)':>12}{'vs dense':>10}")
    for r in data['two_level']:
        two_level_bytes = r['macro_bytes'] + r['micro_bytes']
        blocks_str = f"{r['active_blocks']}/{r['total_blocks']} ({r['active_blocks'] / r['total_blocks'] * 100:.1f}%)"
        print(f"{r['block_size']:<12}{blocks_str:<28}{kib(r['macro_bytes']):>12.2f}{mib(r['micro_bytes']):>12.2f}"
              f"{mib(two_level_bytes):>12.2f}{data['pt_bytes'] / two_level_bytes:>9.2f}x")

    print(f"\n-- padded (block_size+1)^3 blocks --")
    print(f"{'block_size':<12}{'active_blocks':<28}{'macro (KiB)':>12}{'micro (MiB)':>12}"
          f"{'total (MiB)':>12}{'vs dense':>10}")
    for r in data['two_level_padded']:
        padded_bytes = r['macro_bytes'] + r['micro_bytes']
        blocks_str = f"{r['active_blocks']}/{r['total_blocks']} ({r['active_blocks'] / r['total_blocks'] * 100:.1f}%)"
        print(f"{r['block_size']:<12}{blocks_str:<28}{kib(r['macro_bytes']):>12.2f}{mib(r['micro_bytes']):>12.2f}"
              f"{mib(padded_bytes):>12.2f}{data['pt_bytes'] / padded_bytes:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="cloud_356")
    args = parser.parse_args()

    data = compute_sizes(args.volume)
    print_sizes(data)


if __name__ == "__main__":
    main()
