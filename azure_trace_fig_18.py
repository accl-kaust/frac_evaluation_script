import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob
from scipy import stats
import re

import plot_fonts  # Helvetica Neue from fonts/, shared palette and print-size style

# Printed at 0.48\textwidth = 241.9 pt in the paper; its tight-cropped PDF is ~552 pt wide.
pstyle = plot_fonts.paper_style(printed_width_pt=241.9, cropped_width_pt=551.8)

def read_trace_log_data(folder_path):
    """Read trace log files and extract latency data"""
    trace_files = glob.glob(os.path.join(folder_path, "hugepage_thread_*.txt"))
    
    all_data = []
    
    for file_path in trace_files:
        print(f"Reading {file_path}")
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse trace_send and trace_resp pairs
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for trace_send line
            if 'trace_send' in line:
                # Parse trace_send line
                func_match = re.search(r'func=(\d+)', line)
                req_bytes_match = re.search(r'req_bytes=(\d+)', line)
                
                if func_match and req_bytes_match:
                    func_id = int(func_match.group(1))
                    req_bytes = int(req_bytes_match.group(1))
                    
                    # Calculate fragments: req_bytes / 576 gives the fragment count
                    # 576 = 1 fragment (base unit)
                    fragments = (req_bytes) // 512
                    
                    # Look ahead for the latency_us in trace_resp (typically 2 lines later)
                    if i + 2 < len(lines):
                        resp_line = lines[i + 2].strip()
                        latency_match = re.search(r'latency_us=([\d.]+)', resp_line)
                        
                        if latency_match:
                            latency_us = float(latency_match.group(1))
                            
                            all_data.append({
                                'func': func_id,
                                'fragments': fragments,
                                'latency_us': latency_us
                            })
            
            i += 1
    
    print(f"Parsed {len(all_data)} trace entries")
    return pd.DataFrame(all_data)

def tint(hex_color, share):
    """Mix a colour with white: share=1 is the colour itself, smaller values are lighter."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(share * c + (1 - share) for c in (r, g, b))


def create_cdf_plot(data_df, output_file="azure_trace_fig_18.pdf"):
    """Create a single figure with a broken x-axis (0–400 μs and 400–7500 μs)."""
    
    # Get all combinations that exist in the data
    combinations = data_df.groupby(['func', 'fragments']).size().reset_index()
    combinations = [(row['func'], row['fragments']) for _, row in combinations.iterrows()]
    combinations.sort()
    
    print(f"Found {len(combinations)} function-fragment combinations:")
    for func, frag in combinations:
        count = len(data_df[(data_df['func'] == func) & (data_df['fragments'] == frag)])
        print(f"  Func {func} (Frag {frag}): {count} samples")
    
    # Broken x-axis with two panels sharing y-axis; size modeled after tail_latency_cdf.py (~8x2 per panel)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, figsize=(8, 2.8))
    
    # Function name mapping
    func_names = {
        1: 'Top-K',
        2: 'CNN', 
        3: 'Logit',
        5: 'Norm'
    }
    
    # Color mapping for each function type
    func_colors = {  # category colours, same order as fig 14: Top K, Logit, Norm, CNN
        1: plot_fonts.CATEGORY_PALETTE[0],  # Top K
        3: plot_fonts.CATEGORY_PALETTE[1],  # Logit
        5: plot_fonts.CATEGORY_PALETTE[2],  # Norm
        2: plot_fonts.CATEGORY_PALETTE[3],  # CNN
    }
    # Line style encodes the fragment count (same for every accelerator), colour the
    # accelerator. CNN has a single size (48 fragments) and is drawn solid.
    frag_styles = {1: '-', 2: '--', 4: '-.', 8: ':', 48: '-'}

    # Get unique function types, and plot separate lines per fragment with same color
    unique_funcs = sorted(set([func for func, frag in combinations]))

    # Collect legend entries per function for custom multi-column legend
    legend_entries_per_func = {}

    for func in unique_funcs:
        func_name = func_names.get(func, f'Func {func}')
        color = func_colors.get(func, '#000000')

        frag_values = sorted(data_df.loc[data_df['func'] == func, 'fragments'].unique())

        legend_entries_per_func[func] = []
        for frag in frag_values:
            subset = data_df[(data_df['func'] == func) & (data_df['fragments'] == frag)]
            if len(subset) == 0:
                continue

            latencies = subset['latency_us'].values
            sorted_latencies = np.sort(latencies)
            n = len(sorted_latencies)
            cdf_values = np.arange(1, n + 1) / n

            label = f'{func_name} {frag} Frag'
            style = frag_styles.get(frag, '-')

            # Plot on both axes so legend can be unified
            ax_left.plot(sorted_latencies, cdf_values, label=label, color=color,
                         linestyle=style, linewidth=pstyle.curve)
            ax_right.plot(sorted_latencies, cdf_values, label=label, color=color,
                          linestyle=style, linewidth=pstyle.curve)
            legend_entries_per_func[func].append(
                (Line2D([0], [0], color=color, linestyle=style, lw=pstyle.curve), label)
            )

    # Configure broken x-axis: left shows 0–50 (Top K / Logit / Norm), right shows the
    # CNN cluster. The right window is derived from the data so a re-run of the trace
    # (which shifts CNN latency by hundreds of μs) is not clipped out of view: a fixed
    # 40 μs wide window whose lower edge is 15 μs below the CNN minimum, rounded down
    # to a multiple of 5. The CNN rise then sits ~40 % into the panel, leaving the
    # left part of the panel empty for the legend that straddles the axis cut.
    # For the original trace (min ≈ 8836 μs) this gives 8820–8860.
    ax_left.set_xlim(0, 50)
    ax_left.set_xticks(np.arange(0, 51, 10))
    right_latencies = data_df.loc[data_df['func'] == 2, 'latency_us']
    if right_latencies.empty:
        right_latencies = data_df.loc[data_df['latency_us'] > 50, 'latency_us']
    right_lo = int(np.floor((right_latencies.min() - 15) / 5) * 5)
    right_hi = right_lo + 40
    ax_right.set_xlim(right_lo, right_hi)
    ax_right.set_xticks(np.arange(int(np.ceil(right_lo / 10) * 10), right_hi + 1, 10))
    print(f"Right panel x-range: {right_lo}–{right_lo + 40} μs "
          f"(CNN min {right_latencies.min():.1f}, p99 {right_latencies.quantile(0.99):.1f}, "
          f"max {right_latencies.max():.1f})")

    # Hide the spines between ax_left and ax_right
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_left.yaxis.tick_left()
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position('right')

    # Diagonal cut marks
    d = .015
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=pstyle.edge)
    ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)

    # Labels, grid, and legend
    # Use a single centered x-axis label for both panels, positioned closer to the axes
    fig.supxlabel('Latency (μs)', fontsize=pstyle.font, y=0.08, fontweight='medium')
    ax_left.set_ylabel('CDF', fontsize=pstyle.font, fontweight='medium')
    # Grid: vertical dashed lines like the reference. Drawn opaque in the grey that the
    # previous 70 % alpha produced on white (0.7 x #b0b0b0 + 0.3 x white = #c8c8c8), so
    # the look is unchanged but the PDF carries no transparency (ACM print rule).
    ax_left.grid(True, linestyle='--', color='#c8c8c8', alpha=1.0, axis='x')
    ax_right.grid(True, linestyle='--', color='#c8c8c8', alpha=1.0, axis='x')
    # Ticks: size
    ax_left.tick_params(axis='both', which='major', labelsize=pstyle.font)
    ax_right.tick_params(axis='both', which='major', labelsize=pstyle.font)
    for tick in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        tick.set_fontweight('medium')
    for tick in ax_right.get_xticklabels() + ax_right.get_yticklabels():
        tick.set_fontweight('medium')
    
    # Two-column legend: left column = accelerator (colour, solid handle), right column
    # = fragment count (line style, black handle). CNN is annotated with its fixed size.
    preferred_order = [1, 3, 5, 2]
    funcs_order = [f for f in preferred_order if f in legend_entries_per_func]
    acc_handles, acc_labels = [], []
    for func in funcs_order:
        func_name = func_names.get(func, f'Func {func}')
        color = func_colors.get(func, '#000000')
        label = f'{func_name} (48 Frag)' if func == 2 else func_name
        acc_handles.append(Line2D([0], [0], color=color, linestyle='-', lw=pstyle.curve))
        acc_labels.append(label)
    frag_present = sorted(f for f in data_df['fragments'].unique() if f in (1, 2, 4, 8))
    frag_handles = [Line2D([0], [0], color='black', linestyle=frag_styles[f], lw=pstyle.curve)
                    for f in frag_present]
    frag_labels = [f'{f} Frag' for f in frag_present]
    # matplotlib fills legend columns top-to-bottom, so interleave by row count
    nrow = max(len(acc_handles), len(frag_handles))
    blank = Line2D([0], [0], color='none')
    acc_handles += [blank] * (nrow - len(acc_handles)); acc_labels += [''] * (nrow - len(acc_labels))
    frag_handles += [blank] * (nrow - len(frag_handles)); frag_labels += [''] * (nrow - len(frag_labels))
    all_handles = acc_handles + frag_handles
    all_labels = acc_labels + frag_labels

    legend = fig.legend(
        all_handles,
        all_labels,
        loc='center',
        bbox_to_anchor=(0.5, 0.5),
        ncol=2,
        prop={'size': pstyle.font, 'weight': 'medium'},
        labelspacing=0.4,
        columnspacing=1.2,
        framealpha=1.0,
        edgecolor='black',   # black 0.6 pt frame instead of matplotlib's light grey
        handlelength=2.6,    # long enough to show two periods of the dash-dot pattern
        handletextpad=0.5
    )

    plt.tight_layout()

    pos_l, pos_r = ax_left.get_position(), ax_right.get_position()
    legend_cx = (pos_l.x1 + pos_r.x0) / 2 - 0.045  # nudged left: two-column legend must clear the CNN rise
    legend_cy = pos_l.y0 + 0.45 * pos_l.height
    legend.set_bbox_to_anchor((legend_cx, legend_cy), transform=fig.transFigure)

    # Check the legend against the data on both sides of the cut.
    fig.canvas.draw()
    # matplotlib < 3.5 needs the renderer passed explicitly
    bb = legend.get_window_extent(fig.canvas.get_renderer())
    left_edge_us = ax_left.transData.inverted().transform((bb.x0, 0))[0]
    right_edge_us = ax_right.transData.inverted().transform((bb.x1, 0))[0]
    top_cdf = ax_left.transData.inverted().transform((0, bb.y1))[1]
    left_funcs = data_df[data_df['func'] != 2]
    # Latest point on the left panel that is still below the legend's top edge
    left_rise_end = left_funcs['latency_us'].quantile(top_cdf)
    print(f"Legend spans {left_edge_us:.1f} μs (left panel) .. {right_edge_us:.1f} μs (right panel), "
          f"top at CDF {top_cdf:.2f}; left curves reach CDF {top_cdf:.2f} by {left_rise_end:.1f} μs, "
          f"CNN starts at {right_latencies.min():.1f} μs")
    if left_rise_end > left_edge_us or right_edge_us > right_latencies.min():
        print("WARNING: legend overlaps a curve; adjust legend_cy or the right-panel window")

    # Save the plot as PDF only
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    pstyle.report(output_file)
    print(f"CDF plot saved as {output_file}")
    
    plt.close()  # Close the figure to free memory

def main():
    # Path to the Azure trace folder
    data_folder = "data/azure_trace"
    
    print("Reading trace log files...")
    data_df = read_trace_log_data(data_folder)
    
    print(f"Total data points: {len(data_df)}")
    print(f"Function IDs found: {sorted(data_df['func'].unique())}")
    print(f"Fragment counts found: {sorted(data_df['fragments'].unique())}")
    
    # Print summary statistics
    print("\nSummary by function-fragment combination:")
    summary = data_df.groupby(['func', 'fragments']).agg({
        'latency_us': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)
    print(summary)
    
    # Create CDF plot
    print("\nCreating CDF plot...")
    create_cdf_plot(data_df)

if __name__ == "__main__":
    main()
