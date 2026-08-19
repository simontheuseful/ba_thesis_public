import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(labels, render_time, size_mb, volume_name, out_dir, shape=None, render_time_std=None,
                     title="Grid representation comparison", time_label="Render Time per Frame",
                     file_suffix="", sparsity=None):
    x = np.arange(len(labels))
    width = 0.6

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    shape_suffix = f"  --  shape: {tuple(shape)}" if shape is not None else ""
    sparsity_suffix = f"  --  sparsity: {sparsity * 100:.1f}%" if sparsity is not None else ""
    fig.suptitle(f"{title}  --  volume: {volume_name}{shape_suffix}{sparsity_suffix}", fontsize=16)

    err_kw = dict(yerr=render_time_std, capsize=4, error_kw=dict(ecolor='black', elinewidth=1.2)) \
        if render_time_std is not None else {}
    bars1 = ax1.bar(x, render_time, width, color='#4C72B0', edgecolor='black', **err_kw)
    ax1.set_title(f'{time_label} (mean ± std)' if render_time_std is not None else time_label,
                   fontsize=14, pad=15)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax1.bar_label(bars1, fmt='%.2f', padding=8 if render_time_std is not None else 3, fontsize=10)

    bars2 = ax2.bar(x, size_mb, width, color='#55A868', edgecolor='black')
    ax2.set_title('Memory Footprint', fontsize=14, pad=15)
    ax2.set_ylabel('Size (MB)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax2.bar_label(bars2, fmt='%.1f', padding=3, fontsize=10)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{volume_name}{file_suffix}.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    labels = ['Dense', '2-L (bs=1)', '2-L (bs=4)', '2-L (bs=8)', '2-L (bs=16)', '2-L (bs=32)', 'NanoVDB']
    render_time = [2.368, 6.882, 7.654, 7.795, 7.669, 6.025, 21.098]
    render_time_std = [0.05, 0.15, 0.18, 0.20, 0.17, 0.12, 0.40]
    size_mb = [39.38, 49.62, 12.99, 14.57, 18.68, 25.25, 15.76]

    out_path = plot_comparison(labels, render_time, size_mb, "example", ".", shape=(288, 160, 224, 1),
                                render_time_std=render_time_std)
    print(f"Saved {out_path}")
