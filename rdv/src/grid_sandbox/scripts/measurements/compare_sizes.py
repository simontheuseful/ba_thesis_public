"""
Compares on-disk/on-GPU footprint of the same volume across representations:
dense .pt tensor, two-level sparse grid (macro grid + block pool) swept across
block sizes, and .nvdb.

The reference volume is always cropped to a multiple of 32 (matching the .nvdb
files, see nvdb_converter.py) so dense/.nvdb/active-voxel numbers are fixed and
only the two-level breakdown varies with block size.

--plot saves a bar chart to diagrams/{volume}/{volume}_sizes.pdf.

Usage:
    python compare_sizes.py --volume cloud_865
    python compare_sizes.py --volume cloud_865 --plot
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # utility.py lives one level up

import rdv
from utility import load_pt_volume, create_two_level_grid, tensor_bytes
from plotter_sizes import plot_sizes

DATA_DIR = os.path.join(_HERE, "..", "..", "data")
DIAGRAMS_DIR = os.path.join(_HERE, "..", "..", "diagrams")

THRESHOLD = 1e-4
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

    active_voxels = (vol.abs() > THRESHOLD).sum().item()
    total_voxels = vol.numel()

    two_level = []
    for bs in TWO_LEVEL_BLOCK_SIZES:
        macro_grid, block_pool = create_two_level_grid(rdv.tensor_copy(vol), block_size=bs, threshold=THRESHOLD)
        macro_bytes = tensor_bytes(macro_grid)
        micro_bytes = tensor_bytes(block_pool)
        active_blocks = block_pool.shape[0]
        total_blocks = (D // bs) * (H // bs) * (W // bs)
        two_level.append(dict(
            block_size=bs, macro_bytes=macro_bytes, micro_bytes=micro_bytes,
            active_blocks=active_blocks, total_blocks=total_blocks,
        ))

    return dict(
        volume=volume, shape=(D, H, W, C), pt_bytes=pt_bytes, nvdb_bytes=nvdb_bytes,
        active_voxels=active_voxels, total_voxels=total_voxels, two_level=two_level,
    )


def print_sizes(data):
    print(f"volume={data['volume']} shape={data['shape']} threshold={THRESHOLD}")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="cloud_356")
    parser.add_argument("--plot", action="store_true", help="save a bar chart of the results as a pdf")
    args = parser.parse_args()

    data = compute_sizes(args.volume)
    print_sizes(data)

    if args.plot:
        labels = ["Dense"] + [f"2-L (bs={r['block_size']})" for r in data['two_level']] + ["NanoVDB"]
        macro_mib = [0.0] + [mib(r['macro_bytes']) for r in data['two_level']] + [0.0]
        micro_mib = [mib(data['pt_bytes'])] + [mib(r['micro_bytes']) for r in data['two_level']] + [mib(data['nvdb_bytes'])]
        active_voxels_pct = data['active_voxels'] / data['total_voxels'] * 100
        out_path = os.path.join(DIAGRAMS_DIR, args.volume, f"{args.volume}_sizes.pdf")
        plot_sizes(labels, macro_mib, micro_mib, f"sizes -- volume: {args.volume}", out_path,
                   active_voxels_pct=active_voxels_pct, threshold=THRESHOLD, shape=data['shape'])
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
