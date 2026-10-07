"""Figure 5 as one PDF: accelerator utilization (left) and throughput (right) for immediate
vs reassembled ingestion, side by side.

This replaces the former reassembly_utilization_fig_5a.py and reassembly_throughput_fig_5b.py,
whose PDFs the paper placed in two adjacent 0.24\\textwidth minipages. Like figs 9 and 16
the panels carry titles and share one frameless legend above them. The page is 12 in wide
(twice the 6 in page of each former figure) and is meant to be included at 0.48\\textwidth =
241.9 pt, so every panel, font and line prints at exactly the size the two separate
figures did:

    \\includegraphics[width=0.48\\textwidth]{Figures/reassembly_fig_5.pdf}

Like the former figures, the page is not tight-cropped. Data loading and the per-panel
drawing code are theirs; only the layout is shared.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

import plot_style
from plot_colors import (
    BAR_PALETTE, BAR_INDICES, BAR_SATURATION, BAR_WHITE, BAR_BLACK, toned_palette,
)

# Defaults in plot_colors.py; override here for Figure 5 only.
# Order: Immediate (purple), Reassembled (green). Tone controls range from 0 to 1.
palette_name = BAR_PALETTE
indices = list(BAR_INDICES)
saturation = BAR_SATURATION
white = list(BAR_WHITE)  # [0.0, 0.5]: more white makes each color lighter
black = list(BAR_BLACK)  # [0.15, 0.0]: more black makes each color darker
color_baseline, color_improved = [
    toned_palette([index], palette=palette_name, saturation=saturation,
                  white=white[n], black=black[n])[0]
    for n, index in enumerate(indices)
]

plot_style.apply()
# Printed at 0.48\textwidth = 241.9 pt in the paper; the 12 in wide page is not tight-cropped.
pstyle = plot_style.paper_style(printed_width_pt=241.9, cropped_width_pt=864.0)

# Inputs live in <repo>/Data_simulation and figures go to <repo>/Plot, both resolved
# relative to this file so the script works from any working directory.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_DIR, 'Data_simulation')
PLOT_DIR = os.path.join(_REPO_DIR, 'Plot')
OUTPUT_FILE = os.path.join(PLOT_DIR, 'reassembly_fig_5.pdf')

# Panel titles, as in fig 9 ("JSQ" / "RR"); set to None to drop them.
PANEL_TITLES = ('Accelerator Utilization', 'Throughput')
# Page height in inches. The originals are 4 in tall; the legend row and the titles take
# about 0.3 in at this scale, so 4.3 in keeps the panels at the height of fig 5b's.
PAGE_HEIGHT_IN = 4.3

PACKET_DISTRIBUTION = [1, 2, 4, 8]
distribution_names = ['1', '2', '4', '8']
FRAGMENT_PROCESSING_US = 20   # each fragment keeps the accelerator busy for 20 us


# --------------------------------------------------------------------------- #
# Left panel: utilization, from reassembly_utilization_fig_5a.py
# --------------------------------------------------------------------------- #
UTIL_CSV = os.path.join(DATA_DIR, 'Reassemble/1/linear/1/all_A/{}/RR/RR/Inter_Packet_30/{}/latency.csv')
UTIL_MODES = ['NO_CORE_QUEUE', 'PER_CORE_QUEUE']   # immediate, reassembled


def load_utilization(distribution, mode):
    """Per-request busy share (%) of the accelerator: 20 us x fragments / processing time."""
    file_path = UTIL_CSV.format(mode, distribution)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.array([])
    total_processing_time = pd.read_csv(file_path)['Total Processing Time'].values
    return (FRAGMENT_PROCESSING_US * distribution / total_processing_time) * 100


def draw_utilization(ax):
    positions = np.arange(len(PACKET_DISTRIBUTION)) * 2
    box_width = 0.4
    for i, dist in enumerate(PACKET_DISTRIBUTION):
        immediate = load_utilization(dist, UTIL_MODES[0])
        reassembled = load_utilization(dist, UTIL_MODES[1])
        ax.boxplot(immediate, positions=[positions[i]], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor=color_baseline, edgecolor='black', linewidth=pstyle.edge),
                   whiskerprops=dict(linewidth=pstyle.edge),
                   capprops=dict(linewidth=pstyle.edge),
                   medianprops=dict(color='black', linewidth=pstyle.median),
                   showfliers=False)
        ax.boxplot(reassembled, positions=[positions[i] + box_width + 0.1], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor=color_improved, edgecolor='black', linewidth=pstyle.edge),
                   whiskerprops=dict(linewidth=pstyle.edge),
                   capprops=dict(linewidth=pstyle.edge),
                   medianprops=dict(color='black', linewidth=pstyle.median),
                   showfliers=False)

    ax.set_xlabel('Request Size (Fragments)', fontsize=pstyle.font, fontweight='medium')
    ax.set_ylabel('Utilization (%)', fontsize=pstyle.font, fontweight='medium')
    ax.set_xticks(positions + box_width / 2)
    ax.set_xticklabels(distribution_names, fontsize=pstyle.font, fontweight='medium')
    ax.tick_params(axis='y', labelsize=pstyle.font)
    for label in ax.get_yticklabels():
        label.set_weight('medium')


# --------------------------------------------------------------------------- #
# Right panel: throughput, from reassembly_throughput_fig_5b.py
# --------------------------------------------------------------------------- #
THRPT_CSV = os.path.join(
    DATA_DIR, 'Reassemble_throughput_two_buffer/{}/linear/2/all_A/{}/RR/RR/Inter_Packet_30/{}/latency.csv')
THRPT_MODES = ['NO_CORE_QUEUE', 'REASSEMBLY_BUFFER']   # immediate, reassembled


def load_throughput(distribution, mode):
    """Fragments per second over the whole run (100 requests of `distribution` fragments)."""
    file_path = THRPT_CSV.format(1, mode, distribution)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.nan
    df = pd.read_csv(file_path)
    total_duration_us = df['Total End Time'].max() - df['Total Start Time'].min()
    return 100 * distribution / (total_duration_us / 1_000_000)


def draw_throughput(ax):
    immediate = [load_throughput(d, THRPT_MODES[0]) for d in PACKET_DISTRIBUTION]
    reassembled = [load_throughput(d, THRPT_MODES[1]) for d in PACKET_DISTRIBUTION]

    x = np.arange(len(PACKET_DISTRIBUTION))
    width = 0.3
    ax.bar(x - width / 2, immediate, width, label='Immediate',
           color=color_baseline, edgecolor='black', linewidth=pstyle.edge)
    ax.bar(x + width / 2, reassembled, width, label='Reassembled',
           color=color_improved, edgecolor='black', linewidth=pstyle.edge)

    ax.set_xlabel('Request Size (Fragments)', fontsize=pstyle.font, fontweight='medium')
    ax.set_ylabel('Throughput (Frag/s)', fontsize=pstyle.font, fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels(distribution_names, fontsize=pstyle.font, fontweight='medium')
    ax.tick_params(axis='y', labelsize=pstyle.font)
    for label in ax.get_yticklabels():
        label.set_weight('medium')

    # Scientific notation on the y-axis, exponent in the same font size and weight
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(pstyle.font)
    ax.yaxis.get_offset_text().set_weight('medium')


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, PAGE_HEIGHT_IN))
    draw_utilization(ax_a)
    draw_throughput(ax_b)

    if PANEL_TITLES:
        for ax, title in zip((ax_a, ax_b), PANEL_TITLES):
            ax.set_title(title, fontsize=pstyle.font, fontweight='medium')

    # One frameless legend for both panels (same two series in each), centred above the
    # panel titles as in fig 16. tight_layout is told to leave the legend's row free.
    handles = [
        mpatches.Patch(facecolor=color_baseline, label='Immediate', edgecolor='black', linewidth=pstyle.edge),
        mpatches.Patch(facecolor=color_improved, label='Reassembled', edgecolor='black', linewidth=pstyle.edge),
    ]
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                        prop={'weight': 'medium', 'size': pstyle.font}, columnspacing=1.0, handletextpad=0.5)
    fig.canvas.draw()
    legend_h = legend.get_window_extent(fig.canvas.get_renderer()).height / (fig.get_size_inches()[1] * fig.dpi)
    plt.tight_layout(rect=[0, 0, 1, 1 - legend_h])

    fh = fig.get_size_inches()[1] * 72 * pstyle.scale
    print(f"Panel heights (printed pt): left {ax_a.get_position().height * fh:.1f}, right {ax_b.get_position().height * fh:.1f}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE)
    pstyle.report(OUTPUT_FILE)


if __name__ == '__main__':
    main()
