"""
Bar-chart plotting for measure.py timing results, used by sweep.py --plot.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# yellow, orange, red, magenta, dark blue/purple
PALETTE = ['#f8b333', '#f28800', '#e30713', '#e6007d', '#302782']


def plot_times(labels, means_ms, stds_ms, title, out_path, active_voxels_pct=None, threshold=None, shape=None):
    x = np.arange(len(labels))
    width = 0.6

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 5))
    fig.suptitle(title, fontsize=16)

    subtitle_lines = []
    if shape is not None:
        D, H, W = shape[:3]
        subtitle_lines.append(f'{D}×{H}×{W} (D×H×W)')
    subtitle = 'Time per call (mean ± std)'
    if active_voxels_pct is not None:
        subtitle += f'  --  active voxels: {active_voxels_pct:.1f}%'
        if threshold is not None:
            subtitle += f' (threshold={threshold:.0e})'
    subtitle_lines.append(subtitle)
    ax.set_title('\n'.join(subtitle_lines), fontsize=14, pad=15)

    bars = ax.bar(x, means_ms, width, color=PALETTE[4], edgecolor='black',
                   yerr=stds_ms, capsize=4, error_kw=dict(ecolor='black', elinewidth=1.2))
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.bar_label(bars, fmt='%.2f', padding=8, fontsize=10)

    fig.text(0.5, -0.02, 'bs = block size', ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return out_path
