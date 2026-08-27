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


def _time_subtitle(active_voxels_pct, threshold, shape, prefix='Time per call (mean ± std)'):
    lines = []
    if shape is not None:
        D, H, W = shape[:3]
        lines.append(f'{D}×{H}×{W} (D×H×W)')
    subtitle = prefix
    if active_voxels_pct is not None:
        subtitle += f'  --  active voxels: {active_voxels_pct:.1f}%'
        if threshold is not None:
            subtitle += f' (threshold={threshold:.0e})'
    lines.append(subtitle)
    return '\n'.join(lines)


def plot_times(labels, means_ms, stds_ms, title, out_path, active_voxels_pct=None, threshold=None, shape=None,
               pdf=None):
    x = np.arange(len(labels))
    width = 0.6

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 5))
    fig.suptitle(title, fontsize=16)
    ax.set_title(_time_subtitle(active_voxels_pct, threshold, shape), fontsize=14, pad=15)

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
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_times_paired(group_labels, means_a, stds_a, means_b, stds_b, title, out_path,
                       label_a='without DDA', label_b='with DDA',
                       active_voxels_pct=None, threshold=None, shape=None, pdf=None):
    """One tick per group, two colored bars per group (e.g. non-DDA vs DDA) instead
    of a separate labeled bar for each."""
    x = np.arange(len(group_labels))
    width = 0.38

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(6, len(group_labels) * 1.3), 5))
    fig.suptitle(title, fontsize=16)
    ax.set_title(_time_subtitle(active_voxels_pct, threshold, shape), fontsize=14, pad=15)

    bars_a = ax.bar(x - width / 2, means_a, width, color=PALETTE[4], edgecolor='black', label=label_a,
                     yerr=stds_a, capsize=3, error_kw=dict(ecolor='black', elinewidth=1.0))
    bars_b = ax.bar(x + width / 2, means_b, width, color=PALETTE[1], edgecolor='black', label=label_b,
                     yerr=stds_b, capsize=3, error_kw=dict(ecolor='black', elinewidth=1.0))

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=11)
    ax.bar_label(bars_a, fmt='%.2f', padding=3, fontsize=8)
    ax.bar_label(bars_b, fmt='%.2f', padding=3, fontsize=8)

    ax.legend(loc='upper left', fontsize=10)
    fig.text(0.5, -0.02, 'bs = block size', ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return out_path
