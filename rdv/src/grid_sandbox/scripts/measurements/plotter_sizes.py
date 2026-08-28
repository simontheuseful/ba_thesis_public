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


def plot_sizes_compare(labels, micro_a, macro_a, micro_b, macro_b, label_a, label_b, title, out_path,
                        ref_bars=None, ref_group_label='Reference',
                        active_voxels_pct=None, threshold=None, shape=None, pdf=None):
    """
    One tick per block size, two stacked (macro+micro) bars per tick -- label_a vs label_b
    block_pool storage side by side. ref_bars adds one extra group at the end with a
    standalone bar per fixed-size configuration that doesn't vary by block size (e.g.
    dense, nanovdb): list of (name, total_mib).
    """
    has_ref = bool(ref_bars)
    all_labels = list(labels) + ([ref_group_label] if has_ref else [])
    x = np.arange(len(all_labels))
    width = 0.38
    total_a = [m + u for m, u in zip(macro_a, micro_a)]
    total_b = [m + u for m, u in zip(macro_b, micro_b)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(7, len(all_labels) * 1.5), 5.5))
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

    bs_x = x[:len(labels)]
    ax.bar(bs_x - width / 2, micro_a, width, color=PALETTE[4], edgecolor='black', label=f'micro array ({label_a})', zorder=2)
    ax.bar(bs_x - width / 2, macro_a, width, bottom=micro_a, color=PALETTE[1], edgecolor='black', label=f'macro array ({label_a})', zorder=2)
    ax.bar(bs_x + width / 2, micro_b, width, color=PALETTE[3], edgecolor='black', label=f'micro array ({label_b})', zorder=2)
    ax.bar(bs_x + width / 2, macro_b, width, bottom=micro_b, color=PALETTE[0], edgecolor='black', label=f'macro array ({label_b})', zorder=2)

    for xi, total in zip(bs_x, total_a):
        ax.text(xi - width / 2, total, f'{total:.1f}', ha='center', va='bottom', fontsize=8)
    for xi, total in zip(bs_x, total_b):
        ax.text(xi + width / 2, total, f'{total:.1f}', ha='center', va='bottom', fontsize=8)

    if has_ref:
        m = len(ref_bars)
        ref_width = 0.8 / max(m, 1)
        ref_x = x[len(labels)]
        for i, (name, total_mib) in enumerate(ref_bars):
            offset = (i - (m - 1) / 2) * ref_width
            bar = ax.bar(ref_x + offset, total_mib, ref_width * 0.92, color=PALETTE[i % len(PALETTE)],
                         edgecolor='black', hatch='///', label=name, zorder=2)
            ax.bar_label(bar, fmt='%.1f', padding=2, fontsize=8)

    ax.set_ylabel('Size (MiB)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    fig.text(0.5, -0.02, 'bs = block size', ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return out_path
