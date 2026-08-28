"""
Runs measure.py's measure() across variant/block-size/mode combinations for one volume and
builds one of three fixed comparison diagrams. Each covers one sampling mode and shows every
applicable grid variant: two_level / two_level_dda / two_level_padded / two_level_dda_padded
as grouped bars across block sizes 1-32, with dense / nanovdb / nanovdb_dda (which have no
block size of their own) drawn as horizontal reference lines.

Methods:
    raymarching_compare -- two_level, two_level_dda, two_level_padded, two_level_dda_padded
                   (bs=1,2,4,8,16,32) vs dense/nanovdb/nanovdb_dda -- mode=render
    random_compare -- two_level, two_level_padded (bs=1,2,4,8,16,32) vs dense/nanovdb
                   -- mode=random (point queries; DDA variants take a 6D ray, not a 3D
                   point, so they have no point-query form)
    linear_compare -- same as random_compare -- mode=linear (point queries)

--plot saves the comparison to diagrams/{volume}/{volume}_{method}.pdf.

Usage:
    python sweep.py raymarching_compare --volume cloud_865 --plot
    python sweep.py random_compare --volume cloud_865 --plot
    python sweep.py linear_compare --volume cloud_865 --plot
"""
import argparse
import os

from measure import measure, format_result, THRESHOLD, QUERY_BOX_SHAPE
from plotter import plot_times_compare

_HERE = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS_DIR = os.path.join(_HERE, "..", "..", "diagrams")

BLOCK_SIZES = [1, 2, 4, 8, 16, 32]
NO_BLOCK_SIZE_REF = 32  # crop used for variants without a real block-size concept (dense, nanovdb)


ITERS = 10  # bumped from measure()'s default of 3 -- n=3 was noisy enough to show a spurious
            # bump in at least one series (see disney_cloud_raymarching_compare's original
            # 2-Level-Padded bs=4 point); with n=10 that flattens out as expected.


def run(variant, volume, block_size, mode="render"):
    result = measure(variant, volume, block_size, mode, iters=ITERS)
    print(f"\n{format_result(result)}")
    return result


def _series(variant, label, volume, mode):
    """One bar series: variant measured at every block size. Returns (label, means, stds, results)."""
    results = [run(variant, volume, bs, mode) for bs in BLOCK_SIZES]
    return label, [r["mean_ms"] for r in results], [r["std_ms"] for r in results], results


def _ref(variant, label, volume, mode):
    """One reference line: variant measured once (no real block-size concept). Returns (label, mean_ms, result)."""
    r = run(variant, volume, NO_BLOCK_SIZE_REF, mode)
    return label, r["mean_ms"], r


def raymarching_compare(volume):
    series = [
        _series("two_level", "2-Level", volume, "render"),
        _series("two_level_dda", "2-Level-DDA", volume, "render"),
        _series("two_level_padded", "2-Level-Padded", volume, "render"),
        _series("two_level_dda_padded", "2-Level-DDA-Padded", volume, "render"),
    ]
    refs = [
        _ref("dense", "Dense", volume, "render"),
        _ref("nanovdb", "NanoVDB", volume, "render"),
        _ref("nanovdb_dda", "NanoVDB-DDA", volume, "render"),
    ]
    return dict(series=series, refs=refs, first=series[0][3][0])


def _query_compare(volume, mode):
    series = [
        _series("two_level", "2-Level", volume, mode),
        _series("two_level_padded", "2-Level-Padded", volume, mode),
    ]
    refs = [
        _ref("dense", "Dense", volume, mode),
        _ref("nanovdb", "NanoVDB", volume, mode),
    ]
    return dict(series=series, refs=refs, first=series[0][3][0])


def random_compare(volume):
    return _query_compare(volume, "random")


def linear_compare(volume):
    return _query_compare(volume, "linear")


METHODS = {
    "raymarching_compare": raymarching_compare,
    "random_compare": random_compare,
    "linear_compare": linear_compare,
}


def plot_rows(method, volume, data, pdf=None):
    """Plots one method's data (as returned by METHODS[method](volume)) and saves to
    diagrams/{volume}/{volume}_{method}.pdf. Returns the saved path. If pdf (a matplotlib
    PdfPages) is given, the figure is also appended to it."""
    group_labels = [f"bs={bs}" for bs in BLOCK_SIZES]
    plot_series = [(label, means, stds) for label, means, stds, _ in data["series"]]
    ref_bars = [(label, mean_ms, r["std_ms"]) for label, mean_ms, r in data["refs"]]
    first = data["first"]
    active_voxels_pct = first["active_voxels"] / first["total_voxels"] * 100
    out_path = os.path.join(DIAGRAMS_DIR, volume, f"{volume}_{method}.pdf")

    dims_label = None
    if method in ("random_compare", "linear_compare"):
        bd, _, _ = QUERY_BOX_SHAPE
        plot_title = method.replace("_compare", "")
        dims_label = (f"{bd}³ points sampled from [-1,1]³" if method == "random_compare"
                       else f"{bd}³ points sampled linearly from [-1,1]³")
    else:
        plot_title = f"raymarching -- volume: {volume}"

    plot_times_compare(group_labels, plot_series, plot_title, out_path,
                        ref_bars=ref_bars, active_voxels_pct=active_voxels_pct, threshold=THRESHOLD,
                        shape=first["shape"], dims_label=dims_label, pdf=pdf)

    print(f"\nSaved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=list(METHODS))
    parser.add_argument("--volume", default="cloud_356")
    parser.add_argument("--plot", action="store_true", help="save the comparison as a pdf")
    args = parser.parse_args()

    data = METHODS[args.method](args.volume)

    if args.plot:
        plot_rows(args.method, args.volume, data)


if __name__ == "__main__":
    main()
