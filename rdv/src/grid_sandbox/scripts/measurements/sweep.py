"""
Runs measure.py's measure() across a batch of variant/block-size/mode combinations
for one volume, one method at a time, printing each run's result (same console
format as measure.py itself).

Methods:
    basic       -- dense, two_level (bs=1,2,4,8,16,32), nanovdb -- render mode
    random      -- same combos as basic, --mode random (point queries)
    linear      -- same combos as basic, --mode linear (point queries)
    dda         -- two_level_dda (bs=1,2,4,8,16,32), nanovdb_dda -- render mode
    dda_compare -- two_level vs two_level_dda for each block size (bs=1,2,4,8,16,32),
                   then nanovdb vs nanovdb_dda -- render mode

dense/nanovdb have no block-size of their own, so their single run uses
block-size 32 -- the crop that exactly matches the .nvdb files (see
nvdb_converter.py), for maximum consistency with the two_level_dda=32 row.

--plot saves a bar chart of the method's results to
diagrams/{volume}/{volume}_{method}.pdf.

Usage:
    python sweep.py basic --volume cloud_865
    python sweep.py dda --volume cloud_356 --plot
    python sweep.py dda_compare --volume cloud_356 --plot
"""
import argparse
import os

from measure import measure, format_result, THRESHOLD
from plotter import plot_times, plot_times_paired

_HERE = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS_DIR = os.path.join(_HERE, "..", "..", "diagrams")

BLOCK_SIZES = [1, 2, 4, 8, 16, 32]
NO_BLOCK_SIZE_REF = 32  # crop used for variants without a real block-size concept (dense, nanovdb)


def run(label, variant, volume, block_size, mode="render"):
    result = measure(variant, volume, block_size, mode)
    print(f"\n{format_result(result)}")
    return label, result


def basic(volume, mode="render"):
    rows = [run("Dense", "dense", volume, NO_BLOCK_SIZE_REF, mode)]
    for bs in BLOCK_SIZES:
        rows.append(run(f"2-L (bs={bs})", "two_level", volume, bs, mode))
    rows.append(run("NanoVDB", "nanovdb", volume, NO_BLOCK_SIZE_REF, mode))
    return rows


def random_(volume):
    return basic(volume, mode="random")


def linear(volume):
    return basic(volume, mode="linear")


def dda(volume):
    rows = [run(f"2-L-DDA (bs={bs})", "two_level_dda", volume, bs) for bs in BLOCK_SIZES]
    rows.append(run("NanoVDB-DDA", "nanovdb_dda", volume, NO_BLOCK_SIZE_REF))
    return rows


def dda_compare(volume):
    rows = []
    for bs in BLOCK_SIZES:
        rows.append(run(f"2-L (bs={bs})", "two_level", volume, bs))
        rows.append(run(f"2-L-DDA (bs={bs})", "two_level_dda", volume, bs))
    rows.append(run("NanoVDB", "nanovdb", volume, NO_BLOCK_SIZE_REF))
    rows.append(run("NanoVDB-DDA", "nanovdb_dda", volume, NO_BLOCK_SIZE_REF))
    return rows


METHODS = {
    "basic": basic,
    "random": random_,
    "linear": linear,
    "dda": dda,
    "dda_compare": dda_compare,
}


def plot_rows(method, volume, rows, pdf=None):
    """Plots one method's rows (as returned by METHODS[method](volume)) and saves
    to diagrams/{volume}/{volume}_{method}.pdf. Returns the saved path. If pdf (a
    matplotlib PdfPages) is given, the figure is also appended to it."""
    first = rows[0][1]
    active_voxels_pct = first["active_voxels"] / first["total_voxels"] * 100
    out_path = os.path.join(DIAGRAMS_DIR, volume, f"{volume}_{method}.pdf")
    plot_title = f"{method} -- volume: {volume}"

    if method == "dda_compare":
        # rows alternate (non-DDA, DDA) per group -- see dda_compare() above
        no_dda_rows, dda_rows = rows[0::2], rows[1::2]
        group_labels = [label for label, _ in no_dda_rows]
        means_a = [r["mean_ms"] for _, r in no_dda_rows]
        stds_a = [r["std_ms"] for _, r in no_dda_rows]
        means_b = [r["mean_ms"] for _, r in dda_rows]
        stds_b = [r["std_ms"] for _, r in dda_rows]
        plot_times_paired(group_labels, means_a, stds_a, means_b, stds_b, plot_title, out_path,
                           label_a="without DDA", label_b="with DDA",
                           active_voxels_pct=active_voxels_pct, threshold=THRESHOLD, shape=first["shape"], pdf=pdf)
    else:
        labels = [label for label, _ in rows]
        means_ms = [r["mean_ms"] for _, r in rows]
        stds_ms = [r["std_ms"] for _, r in rows]
        plot_times(labels, means_ms, stds_ms, plot_title, out_path,
                   active_voxels_pct=active_voxels_pct, threshold=THRESHOLD, shape=first["shape"], pdf=pdf)

    print(f"\nSaved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=list(METHODS))
    parser.add_argument("--volume", default="cloud_356")
    parser.add_argument("--plot", action="store_true", help="save a bar chart of the results as a pdf")
    args = parser.parse_args()

    rows = METHODS[args.method](args.volume)

    if args.plot:
        plot_rows(args.method, args.volume, rows)


if __name__ == "__main__":
    main()
