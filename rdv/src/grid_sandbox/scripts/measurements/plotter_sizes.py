"""
Bar-chart plotting for compare_sizes.py results, used by compare_sizes.py --plot.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# yellow, orange, red, magenta, dark blue/purple
PALETTE = ['#f8b333', '#f28800', '#e30713', '#e6007d', '#302782']


def plot_sizes(labels, macro_mib, micro_mib, title, out_path, active_voxels_pct=None, threshold=None, shape=None):
    """
    macro_mib[i] / micro_mib[i]: stacked components for bar i. For bars with no
    macro/micro split (dense, .nvdb), pass macro_mib[i] = 0 -- the bar then
    collapses to a single color.
    """
    x = np.arange(len(labels))
    width = 0.6
    total_mib = [m + u for m, u in zip(macro_mib, micro_mib)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 5))
    fig.suptitle(title, fontsize=16)

    subtitle_lines = []
    if shape is not None:
        D, H, W = shape[:3]
        subtitle_lines.append(f'{D}×{H}×{W} (D×H×W)')
    subtitle = 'Size (MiB)'
    if active_voxels_pct is not None:
        subtitle += f'  --  active voxels: {active_voxels_pct:.1f}%'
        if threshold is not None:
            subtitle += f' (threshold={threshold:.0e})'
    subtitle_lines.append(subtitle)
    ax.set_title('\n'.join(subtitle_lines), fontsize=14, pad=15)

    ax.bar(x, micro_mib, width, color=PALETTE[4], edgecolor='black', label='data (dense / block pool / .nvdb)')
    ax.bar(x, macro_mib, width, bottom=micro_mib, color=PALETTE[1], edgecolor='black', label='macro grid')

    ax.set_ylabel('Size (MiB)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)

    for xi, total in zip(x, total_mib):
        ax.text(xi, total, f'{total:.2f}', ha='center', va='bottom', fontsize=10)

    ax.legend(loc='upper right', fontsize=10)
    fig.text(0.5, -0.02, 'bs = block size', ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return out_path
