import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob
from scipy import stats
import re

import plot_fonts  # noqa: F401  (Helvetica Neue from fonts/, medium weight everywhere)

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
    func_colors = {
        1: '#00429d',  # Dark Blue for Top K (legend & Frag 1)
        2: '#d62728',  # Red for CNN
        3: '#ff7f0e',  # Orange for Logit
        5: '#006400'   # Dark Green for Normalization
    }
    
    # Get unique function types, and plot separate lines per fragment with same color
    unique_funcs = sorted(set([func for func, frag in combinations]))

    # Collect legend entries per function for custom multi-column legend
    legend_entries_per_func = {}

    for func in unique_funcs:
        func_name = func_names.get(func, f'Func {func}')
        color = func_colors.get(func, '#000000')
        
        frag_values = sorted(data_df.loc[data_df['func'] == func, 'fragments'].unique())
        # Compute alpha sequence (same color, lighter for larger fragments)
        if len(frag_values) > 1:
            alpha_values = []
            for frag in frag_values:
                if func == 1:  # Top K special case: frag 1 dark blue, others light blue
                    alpha_values.append(1.0 if frag == 1 else 0.4)
                else:
                    alpha_values.append(1.0 if frag == frag_values[0] else 0.6)
        else:
            alpha_values = [1.0]

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
            alpha = float(alpha_values[idx])
            
            # Plot on both axes so legend can be unified
            ax_left.plot(sorted_latencies, cdf_values, label=label, color=color, alpha=alpha, linewidth=2)
            ax_right.plot(sorted_latencies, cdf_values, label=label, color=color, alpha=alpha, linewidth=2)

            # Create proxy handle for legend (solid line with appropriate alpha)
            legend_entries_per_func[func].append(
                (Line2D([0], [0], color=color, lw=2, alpha=alpha), label)
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
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
    ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)

    # Labels, grid, and legend
    # Use a single centered x-axis label for both panels, positioned closer to the axes
    fig.supxlabel('Latency (μs)', fontsize=16, y=0.08, fontweight='medium')
    ax_left.set_ylabel('CDF', fontsize=16, fontweight='medium')
    # Grid: vertical dashed lines like the reference
    ax_left.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax_right.grid(True, linestyle='--', alpha=0.7, axis='x')
    # Ticks: size
    ax_left.tick_params(axis='both', which='major', labelsize=16)
    ax_right.tick_params(axis='both', which='major', labelsize=16)
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
        handle = Line2D([0], [0], color=color, lw=2, alpha=1.0)
        all_handles.append(handle)
        all_labels.append(label)
    
    # Place legend in the middle gap, shifted slightly to the right to reduce overlap
    fig.legend(
        all_handles,
        all_labels,
        loc='center left',
        bbox_to_anchor=(0.52, 0.55),
        ncol=1,
        prop={'size': 13, 'weight': 'medium'},
        labelspacing=0.4,
        framealpha=0.5,
        handlelength=2.4,
        handletextpad=0.6
    )
    
    plt.tight_layout()
    
    # Save the plot as PDF only
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
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
