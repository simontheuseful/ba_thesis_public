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


def _time_subtitle(active_voxels_pct, threshold, shape, dims_label=None, prefix='Time per call (mean ± std)'):
    lines = []
    if dims_label is not None:
        lines.append(dims_label)
    elif shape is not None:
        D, H, W = shape[:3]
        lines.append(f'{D}×{H}×{W} (D×H×W)')
    subtitle = prefix
    if active_voxels_pct is not None:
        subtitle += f'  --  active voxels: {active_voxels_pct:.1f}%'
        if threshold is not None:
            subtitle += f' (threshold={threshold:.0e})'
    lines.append(subtitle)
    return '\n'.join(lines)


def plot_times_compare(group_labels, series, title, out_path,
                        ref_bars=None, ref_group_label='Reference', active_voxels_pct=None,
                        threshold=None, shape=None, dims_label=None, pdf=None):
    """
    group_labels: x-axis groups (e.g. block sizes "bs=1".."bs=32").
    series: list of (name, means_ms, stds_ms), one bar per series per group, legend order.
    ref_bars: list of (name, mean_ms, std_ms) -- fixed-cost configurations with no block-size
              axis (e.g. dense, nanovdb), drawn as their own bars in one extra group at the end.
    """
    n = len(series)
    has_ref = bool(ref_bars)
    all_labels = list(group_labels) + ([ref_group_label] if has_ref else [])
    x = np.arange(len(all_labels))
    width = 0.8 / max(n, 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(7, len(all_labels) * 1.5 + 1), 5.5))
    fig.suptitle(title, fontsize=16)
    ax.set_title(_time_subtitle(active_voxels_pct, threshold, shape, dims_label=dims_label), fontsize=14, pad=15)

    label_fontsize = 8 if n <= 2 else 6
    label_rotation = 0 if n <= 2 else 90
    for i, (name, means, stds) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x[:len(group_labels)] + offset, means, width * 0.92, color=PALETTE[i % len(PALETTE)],
                       edgecolor='black', label=name, yerr=stds, capsize=2,
                       error_kw=dict(ecolor='black', elinewidth=0.8), zorder=2)
        ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=label_fontsize, rotation=label_rotation)

    if has_ref:
        m = len(ref_bars)
        ref_width = 0.8 / max(m, 1)
        ref_x = x[len(group_labels)]
        for i, (name, mean_ms, std_ms) in enumerate(ref_bars):
            offset = (i - (m - 1) / 2) * ref_width
            bar = ax.bar(ref_x + offset, mean_ms, ref_width * 0.92, color=PALETTE[i % len(PALETTE)],
                         edgecolor='black', hatch='///', label=name, yerr=std_ms, capsize=2,
                         error_kw=dict(ecolor='black', elinewidth=0.8), zorder=2)
            ax.bar_label(bar, fmt='%.2f', padding=2, fontsize=label_fontsize, rotation=label_rotation)

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    fig.text(0.5, -0.02, 'bs = block size', ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return out_path
