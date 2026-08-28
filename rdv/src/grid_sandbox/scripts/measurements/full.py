"""
Runs every sweep.py method plus the size comparison for one volume, back to
back, always plotting. The one-shot "everything for this cloud" script.

Besides the usual per-method PDFs (diagrams/{volume}/{volume}_{method}.pdf),
also collects every plot into one combined diagrams/{volume}/{volume}_full.pdf,
one plot per page, in the order the methods ran.

No pauses between measurements. Each measure() call already has its own
warmup to absorb one-time pipeline/JIT costs, and there's no shared state
between different variant/mode runs to contaminate (each builds its own grid
fresh). Earlier profiling in this project also found that some variants (e.g.
fixed-step NanoVDB) show high run-to-run variance even across fully separate,
freshly-launched processes -- about as "paused" as it gets -- so isolation
doesn't appear to fix that noise; it looks intrinsic to the workload rather
than cross-measurement contamination. If a result looks noisy, more --iters
in measure.py is a more effective fix than adding pauses here.

Usage:
    python full.py --volume cloud_865
"""
import argparse
import os

from matplotlib.backends.backend_pdf import PdfPages

from sweep import METHODS, plot_rows, DIAGRAMS_DIR
from compare_sizes import compute_sizes, print_sizes, plot_data as plot_sizes_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True)
    args = parser.parse_args()

    full_path = os.path.join(DIAGRAMS_DIR, args.volume, f"{args.volume}_full.pdf")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with PdfPages(full_path) as pdf:
        for method in METHODS:
            print(f"\n{'=' * 80}\n{method}  --  volume={args.volume}\n{'=' * 80}")
            data = METHODS[method](args.volume)
            plot_rows(method, args.volume, data, pdf=pdf)

        print(f"\n{'=' * 80}\nsizes  --  volume={args.volume}\n{'=' * 80}")
        data = compute_sizes(args.volume)
        print_sizes(data)
        plot_sizes_data(args.volume, data, pdf=pdf)

    print(f"\nSaved {full_path}")


if __name__ == "__main__":
    main()
