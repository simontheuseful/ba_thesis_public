"""
THIS IS ONLY USABLE ON LINUX AND WITH SOME DEPENDENCIES (fvdb).

Converts every .pt volume in a source folder to a .nvdb file (same base name) in an
output folder.

Usage:
    python nvdb_converter.py source output
"""
import argparse
import os

import torch
import fvdb

BLOCK_SIZE = 32  # keep in sync with the max --block-size accepted by measure.py / compare_sizes.py
THRESHOLD = 1e-4


def convert(pt_path: str, nvdb_path: str):
    vol = torch.load(pt_path, map_location="cuda", weights_only=True)
    D, H, W, C = vol.shape
    D, H, W = (D // BLOCK_SIZE) * BLOCK_SIZE, (H // BLOCK_SIZE) * BLOCK_SIZE, (W // BLOCK_SIZE) * BLOCK_SIZE
    vol = vol[:D, :H, :W, :]

    vol_3d = vol.squeeze(-1)
    vol_3d = vol_3d.permute(2, 1, 0).contiguous()  # (D,H,W) -> (W,H,D), matches nanovdb_grid3d.h's ijk convention
    mask = vol_3d > THRESHOLD

    grid = fvdb.Grid.from_dense(dense_dims=vol_3d.shape, mask=mask, device="cuda")
    ijk = grid.ijk
    values = vol_3d[ijk[:, 0], ijk[:, 1], ijk[:, 2]].unsqueeze(-1)

    grid.save_nanovdb(nvdb_path, data=values, name="density")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="source")
    parser.add_argument("output", nargs="?", default="output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pt_files = sorted(f for f in os.listdir(args.source) if f.endswith(".pt"))
    if not pt_files:
        print(f"No .pt files found in {args.source}")
        return

    for fname in pt_files:
        pt_path = os.path.join(args.source, fname)
        nvdb_path = os.path.join(args.output, os.path.splitext(fname)[0] + ".nvdb")
        print(f"{fname} -> {nvdb_path}")
        convert(pt_path, nvdb_path)


if __name__ == "__main__":
    main()