"""Figure 1: latency breakdown of a 4096 B accelerator invocation for gRPC, RDMA and fRAC.

The numbers are the measured medians, typed in below; their sources are the two summaries
in data/invocation_breakdown/ (gRPC and RDMA) and the fig 13 run for fRAC. Output:
Figures/invocation_breakdown_4096B.pdf, the file name the paper includes at
0.95\\columnwidth (228 pt). The 9 x 4.5 in page is saved without a tight crop, so
the printed size is fixed and the shared print-size style is exact.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_fonts  # Helvetica Neue from fonts/, shared palette and print-size style

# Printed at 0.95\columnwidth = 228 pt in the paper; the page is 9 in = 648 pt wide.
pstyle = plot_fonts.paper_style(printed_width_pt=228.0, cropped_width_pt=648.0)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures',
                        'invocation_breakdown_4096B.pdf')


def plot_invocation_breakdown():
    # Data for 4096 Bytes Payload

    # 1. gRPC + Host FPGA (data/invocation_breakdown/grpc_4096B.csv: per-request client RTT,
    #    XRT run and host<->FPGA copy times)
    grpc_total = 837.533
    grpc_xrt_run = 71.0
    grpc_h2d = 69
    grpc_d2h = 39
    grpc_dma = grpc_h2d + grpc_d2h                            # 106 us (sync host<->FPGA)
    grpc_control = grpc_xrt_run                                # 84 us (XRT kernel run)
    grpc_network = grpc_total - grpc_dma - grpc_control        # 843.857 us

    # 2. Our RDMA + Host FPGA experiment (data/invocation_breakdown/rdma_4096B.csv: median RTT,
    #    input, XRT and output times after warm-up)
    # Total RTT = 134.876 us; XRT/FPGA run = 47 us; server CQ wait = 1 us
    our_rdma_total = 86.
    our_rdma_control = 67.789        # XRT/FPGA run + server input CQ wait
    our_rdma_dma = 0.0                       # P2P zero-copy, no host DMA
    our_rdma_network = our_rdma_total - our_rdma_control - our_rdma_dma  # ~86.876

    # 3. fRAC: end-to-end invocation latency for 4096 B requests (2026-10-02 run), the
    #    value quoted in the paper text
    frac_total = 7.8
    frac_network = 7.8
    frac_dma = 0.0
    frac_control = 0.0

    categories = ['gRPC', 'RDMA', 'fRAC (Ours)']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 4.5), gridspec_kw={'height_ratios': [1.2, 2]})
    fig.subplots_adjust(hspace=0.15)

    bar_width = 0.55
    x = np.arange(len(categories))

    # Latency components get their own colours (sand, salmon, green; light -> dark so the
    # stack also reads in greyscale), distinct from the system and workload colours; all
    # three are light enough for black labels.
    color_network, color_dma, color_control = plot_fonts.COMPONENT_PALETTE
    text_network, text_dma, text_control = 'black', 'black', 'black'

    networks = [grpc_network, our_rdma_network, frac_network]
    dmas     = [grpc_dma,     our_rdma_dma,     frac_dma]
    controls = [grpc_control, our_rdma_control, frac_control]
    totals   = [grpc_total,   our_rdma_total,   frac_total]

    def annotate_segment(value, bottom, x_pos, color='white'):
        if value <= 0:
            return

        top = bottom + value
        for ax in (ax2, ax1):
            y_min, y_max = ax.get_ylim()
            visible_bottom = max(bottom, y_min)
            visible_top = min(top, y_max)
            visible_height = visible_top - visible_bottom
            if visible_height <= 0:
                continue

            y = visible_bottom + visible_height / 2
            ax.text(
                x_pos, y, f'{value:.2f} μs',
                ha='center', va='center',
                color=color, fontweight='medium', fontsize=pstyle.font,
                zorder=4
            )
            return

    for ax in [ax1, ax2]:
        for i in range(len(categories)):
            ax.bar(x[i], networks[i], bar_width,
                   label='Network Transfer' if (ax == ax1 and i == 0) else "",
                   color=color_network, edgecolor='black', linewidth=pstyle.edge, zorder=3)
            ax.bar(x[i], dmas[i], bar_width, bottom=networks[i],
                   label='Host-FPGA DMA' if (ax == ax1 and i == 0) else "",
                   color=color_dma, edgecolor='black', linewidth=pstyle.edge, zorder=3)
            ax.bar(x[i], controls[i], bar_width, bottom=networks[i] + dmas[i],
                   label='Accelerator Control' if (ax == ax1 and i == 0) else "",
                   color=color_control, edgecolor='black', linewidth=pstyle.edge, zorder=3)

    # Broken axis: hide the bulk of gRPC network (100–580 µs). The upper panel starts
    # well below the top of the network segment (658 µs) so a clear band of it is
    # visible above the break, and leaves headroom above the RPC bar so the total
    # label does not sit on a gridline.
    ax1.set_ylim(580, 930)
    ax1.set_yticks([600, 700, 800])
    ax2.set_ylim(0, 100)

    for i in range(len(categories) - 1):
        annotate_segment(networks[i], 0, x[i], color=text_network)
        annotate_segment(dmas[i], networks[i], x[i], color=text_dma)
        annotate_segment(controls[i], networks[i] + dmas[i], x[i], color=text_control)

    # Total latency labels stay above each bar.
    ax1.text(x[0], totals[0] + 12, f'{totals[0]:.1f} μs', ha='center', va='bottom',
             fontweight='medium', fontsize=pstyle.font)
    for i in range(1, len(categories)):
        ax2.text(x[i], totals[i] + 3, f'{totals[i]:.2f} μs', ha='center', va='bottom',
                 fontweight='medium', fontsize=pstyle.font)

    # Hide adjoining spines
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False, bottom=False)
    ax2.tick_params(top=False)
    ax2.xaxis.tick_bottom()

    # Diagonal break marks
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=pstyle.edge)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.text(0.04, 0.55, 'Latency (μs)', va='center', rotation='vertical',
             fontweight='medium', fontsize=pstyle.font)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)

    ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    ax1.legend(loc='upper right', framealpha=0.95, fontsize=pstyle.font)

    plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.18)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"Plot saved to {OUT_PATH}")
    pstyle.report(OUT_PATH)


if __name__ == "__main__":
    plot_invocation_breakdown()
