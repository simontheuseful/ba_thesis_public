"""
Visual sanity check for a single grid/raymarching implementation: shows a slice
of the source volume, then the rendered transmittance image.

Merges what used to be check_transmittance.py, check_transmittance_nanovdb.py,
check_transmittance_nanovdb_dda.py and check_transmittance_twolevel_dda.py into
one script selected by --variant. (check_transmittance_twolevel.py is dropped,
not merged: it referenced TwoLevelGrid3DPadding / create_two_level_grid_with_apron,
both removed from grid_implementations.py / utility.py in a prior refactor.)

Usage:
    python check_transmittance.py --variant dense
    python check_transmittance.py --variant nanovdb_dda --volume disney_cloud
"""
import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements"))
from measure import build_view, load_trimmed_volume, VARIANTS, MAX_BLOCK_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="dense")
    parser.add_argument("--volume", default="disney_cloud")
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()

    vol = load_trimmed_volume(args.volume, block_size=MAX_BLOCK_SIZE)
    plt.imshow(vol[:, :, vol.shape[2] // 2, 0].T.cpu())
    plt.gca().axis('off')
    plt.gca().invert_yaxis()
    plt.show()

    view, shape = build_view(args.variant, args.volume, args.block_size)

    with torch.no_grad():
        img = view.capture()

    plt.imshow(img[0].cpu(), vmin=0.0, vmax=1.0, cmap='Blues_r')
    plt.gca().axis('off')
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0.0)
    plt.show()


if __name__ == "__main__":
    main()
