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
                    fragments = (req_bytes - 64) // 512
                    
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
        1: 'Top K',
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
    
    # Get unique function types, and plot separate lines per fragment with same color
    unique_funcs = sorted(set([func for func, frag in combinations]))

    # Collect legend entries per function for custom multi-column legend
    legend_entries_per_func = {}

    for func in unique_funcs:
        func_name = func_names.get(func, f'Func {func}')
        color = func_colors.get(func, '#000000')
        
        frag_values = sorted(data_df.loc[data_df['func'] == func, 'fragments'].unique())
        # Lighter, fully opaque tints of the function colour for larger fragment counts
        # (tint = share of the colour mixed with white; no transparency, so the lines keep
        # their full stroke weight and do not wash out where they cross).
        if len(frag_values) > 1:
            tint_values = []
            for frag in frag_values:
                if func == 1:  # Top K: frag 1 full colour, the others one lighter tint
                    tint_values.append(1.0 if frag == 1 else 0.55)
                else:
                    tint_values.append(1.0 if frag == frag_values[0] else 0.7)
        else:
            tint_values = [1.0]

        legend_entries_per_func[func] = []
        for idx, frag in enumerate(frag_values):
            subset = data_df[(data_df['func'] == func) & (data_df['fragments'] == frag)]
            if len(subset) == 0:
                continue
            
            latencies = subset['latency_us'].values
            sorted_latencies = np.sort(latencies)
            n = len(sorted_latencies)
            cdf_values = np.arange(1, n + 1) / n
            
            label = f'{func_name} {frag} Frag'
            line_color = tint(color, float(tint_values[idx]))
            
            # Plot on both axes so legend can be unified
            ax_left.plot(sorted_latencies, cdf_values, label=label, color=line_color, linewidth=pstyle.line)
            ax_right.plot(sorted_latencies, cdf_values, label=label, color=line_color, linewidth=pstyle.line)

            # Create proxy handle for legend (solid line in the same tint)
            legend_entries_per_func[func].append(
                (Line2D([0], [0], color=line_color, lw=pstyle.line), label)
            )

    # Configure broken x-axis: left shows 0–50, right shows 8800+
    ax_left.set_xlim(0, 50)
    ax_right.set_xlim(8820, 8860)

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
    # Grid: vertical dashed lines like the reference
    ax_left.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax_right.grid(True, linestyle='--', alpha=0.7, axis='x')
    # Ticks: size
    ax_left.tick_params(axis='both', which='major', labelsize=pstyle.font)
    ax_right.tick_params(axis='both', which='major', labelsize=pstyle.font)
    for tick in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        tick.set_fontweight('medium')
    for tick in ax_right.get_xticklabels() + ax_right.get_yticklabels():
        tick.set_fontweight('medium')
    
    # Build simplified legend with grouped fragments per function
    all_handles = []
    all_labels = []
    preferred_order = [1, 2, 3, 5]
    funcs_order = [f for f in preferred_order if f in legend_entries_per_func]
    if not funcs_order:
        funcs_order = sorted(legend_entries_per_func.keys())
    
    for func in funcs_order:
        func_name = func_names.get(func, f'Func {func}')
        color = func_colors.get(func, '#000000')
        
        # Get all fragment counts for this function
        frag_values = sorted(data_df.loc[data_df['func'] == func, 'fragments'].unique())
        
        # Create a single legend entry per function with fragment info
        if len(frag_values) > 1:
            frag_str = ', '.join([str(f) for f in frag_values])
            label = f'{func_name} ({frag_str} Frag)'
        else:
            label = f'{func_name} ({frag_values[0]} Frag)'
        
        # Use solid line with the function's color for legend
        handle = Line2D([0], [0], color=color, lw=pstyle.line, alpha=1.0)
        all_handles.append(handle)
        all_labels.append(label)
    
    # Place legend in the middle gap, shifted slightly to the right to reduce overlap
    fig.legend(
        all_handles,
        all_labels,
        loc='center left',
        bbox_to_anchor=(0.52, 0.55),
        ncol=1,
        prop={'size': pstyle.font, 'weight': 'medium'},
        labelspacing=0.4,
        framealpha=0.5,
        handlelength=2.4,
        handletextpad=0.6
    )
    
    plt.tight_layout()
    
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
