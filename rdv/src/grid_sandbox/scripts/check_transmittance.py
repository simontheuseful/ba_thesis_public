"""
Usage:
    python check_transmittance.py --variant dense
    python check_transmittance.py --variant nanovdb_dda --volume disney_cloud
    python check_transmittance.py --variant nanovdb_dda --volume disney_cloud --loop
"""
import argparse
import os
import sys
import time
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements"))
from measure import build_view, load_trimmed_volume, VARIANTS, MAX_BLOCK_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="dense")
    parser.add_argument("--volume", default="disney_cloud")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--loop", action="store_true", help="Loop captures continuously for profiling (e.g. Nsight)")
    args = parser.parse_args()

    vol = load_trimmed_volume(args.volume, block_size=MAX_BLOCK_SIZE)

    if not args.loop:
        plt.imshow(vol[:, :, vol.shape[2] // 2, 0].T.cpu())
        plt.gca().axis('off')
        plt.gca().invert_yaxis()
        plt.show()

    view, shape = build_view(args.variant, args.volume, args.block_size)

    with torch.no_grad():
        if args.loop:
            print(f"Running capture loop for {args.variant} ({args.volume})... Press Ctrl+C to exit.")
            try:
                while True:
                    _ = view.capture()
            except KeyboardInterrupt:
                print("\nLoop stopped.")
        else:
            img = view.capture()
            plt.imshow(img[0].cpu(), vmin=0.0, vmax=1.0, cmap='gray_r')
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            plt.tight_layout(pad=0.0)
            plt.show()


if __name__ == "__main__":
    main()