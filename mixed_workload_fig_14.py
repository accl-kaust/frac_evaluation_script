import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.grid_helper_curvelinear as grid_helper_curvelinear
from matplotlib import gridspec

# Use medium text styling throughout the figure.
import plot_fonts  # noqa: F401  (Helvetica Neue from fonts/, medium weight everywhere)

# Function mapping - reordered to Top-K, Logit, Norm, CNN
function_ids = [1, 3, 5, 2]  # Top-K, Logit, Norm, CNN
function_names = ['Top-K', 'Logit', 'Norm', 'CNN']  # Shorter labels for x-axis
function_legend_names = ['Top-K', 'Logit', 'Norm', 'CNN']  # Full names for legend
# Replace with color-blind friendly palette (Okabe-Ito color scheme)
function_colors = ['#003355', '#E67300', '#006B52', '#E6A3D0']  # Blue, Vermilion, Bluish green, Reddish purple
# Add hatching patterns for grayscale distinguishability
#function_hatches = ['//', '\\\\', '||', '++']  # Diagonal, reverse diagonal, vertical, cross patterns

class LogProcessor:
    def __init__(self, file_path, function_id, filter_initial_seconds=0):
        self.file_path = file_path
        self.function_id = function_id
        self.data = []
        self.filter_initial_seconds = filter_initial_seconds
        
        # Regular expressions to parse log lines
        self.data_pattern = re.compile(
            r"(\d+)\s+RR\s+\.(\d+)\s+min=(\d+\.\d+)us\s+avg=(\d+\.\d+)us\s+max=(\d+\.\d+)us\s+"
            r"read\(Gbits/sec\)=(\d+\.\d+)\s+write\(Gbits/sec\)=(\d+\.\d+)\s+count=(\d+)"
        )
    
    def parse_log(self):
        # Parse the log file for latency data
        try:
            with open(self.file_path, 'r') as file:
                print(f"Processing file: {self.file_path}")
                for line in file:
                    match = self.data_pattern.search(line)
                    if match:
                        second, thread, min_latency, avg_latency, max_latency, read_tp, write_tp, count = match.groups()
                        time = int(second)
                        if time >= self.filter_initial_seconds:  # Filter data based on specified seconds
                            self.data.append(float(avg_latency))
                
                print(f"Extracted {len(self.data)} latency data points")
                return len(self.data) > 0
        except Exception as e:
            print(f"Error processing file {self.file_path}: {e}")
            return False

def get_log_files_for_function(directory, function_id):
    """Get all log files for a specific function in a directory"""
    files = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return files
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.log') and f'f_{function_id}' in file_name:
            files.append(os.path.join(directory, file_name))
    
    return sorted(files)

def extract_latencies_for_function(directory, function_id, filter_initial_seconds=0):
    """Extract all latencies for a specific function from logs in a directory"""
    all_latencies = []
    files = get_log_files_for_function(directory, function_id)
    
    for file_path in files:
        processor = LogProcessor(file_path, function_id, filter_initial_seconds)
        if processor.parse_log():
            all_latencies.extend(processor.data)
    
    return np.array(all_latencies)

def create_mixed_figure():
    """Create a mixed figure with single workloads on the left and mixed workloads on the right"""
    # Directories
    single_dir = "data/mixed_workload/isolated"
    mixed_dir = "data/mixed_workload/mixed"
    
    # Check if directories exist
    if not os.path.exists(single_dir):
        print(f"Single workload directory {single_dir} does not exist!")
        return
    if not os.path.exists(mixed_dir):
        print(f"Mixed workload directory {mixed_dir} does not exist!")
        return
    
    # Collect latencies for single and mixed workloads
    latencies_single = {}
    latencies_mixed = {}
    
    for i, function_id in enumerate(function_ids):
        print(f"\nProcessing {function_names[i]} (function_{function_id})...")
        
        # Extract latencies for single workloads - filter first 10 seconds
        print(f"Processing single workloads in {single_dir} - filtering first 10 seconds")
        latencies_single[function_id] = extract_latencies_for_function(single_dir, function_id, filter_initial_seconds=10)
        
        # Extract latencies for mixed workloads - no filtering
        print(f"Processing mixed workloads in {mixed_dir}")
        latencies_mixed[function_id] = extract_latencies_for_function(mixed_dir, function_id)
    
    # Print median values
    print("\nMedians for single workloads:")
    for i, function_id in enumerate(function_ids):
        latencies = latencies_single[function_id]
        if len(latencies) > 0:
            median_val = np.median(latencies)
            print(f"{function_names[i]}: {median_val:.2f} microseconds")
        else:
            print(f"{function_names[i]}: No data")
    
    print("\nMedians for mixed workloads:")
    for i, function_id in enumerate(function_ids):
        latencies = latencies_mixed[function_id]
        if len(latencies) > 0:
            median_val = np.median(latencies)
            print(f"{function_names[i]}: {median_val:.2f} microseconds")
        else:
            print(f"{function_names[i]}: No data")
    
    # Create a figure with broken y-axis
    fig = plt.figure(figsize=(8, 5.2))  # Same output height as fig 13 and fig 15
    
    # Create a gridspec for two subplots, with the top subplot taking up more space for the legend
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 2], left=0.10)  # Add more left margin
    
    # Create the top subplot (for high values)
    ax_top = plt.subplot(gs[0])
    # Create the bottom subplot (for low values)
    ax_bottom = plt.subplot(gs[1])
    
    # Set the y-axis limits for each subplot
    # Top subplot for CNN (high values: ~1000-10000)
    high_min, high_max = 1000, 10000  
    # Bottom subplot for other workloads (low values: ~1-50)
    low_min, low_max = 1, 50
    
    ax_top.set_ylim(high_min, high_max)
    ax_bottom.set_ylim(low_min, low_max)
    
    # Set specific tick locations for top subplot to avoid overlap
    ax_top.set_yticks([8000])  # Use fewer, strategic ticks
    
    # Define bar width
    bar_width = 0.45  # Increase bar width to match assembly_plot_with_dpdk.py
    
    # Define total width and calculate proportions
    total_width = 8
    left_width = total_width * 0.6  # Reduce left width slightly
    right_width = total_width * 0.3  # Maintain right width
    
    # Add more spacing between left and right parts
    separator_width = 1.2
    
    # Calculate positions for left and right parts with right part ending at same position as CNN
    right_end = 7.5  # Define rightmost position
    right_start = right_end - right_width
    left_end = right_start - separator_width
    left_start = left_end - left_width
    
    # Recalculate positions with equal spacing
    left_positions = np.linspace(left_start, left_end, 4)
    right_positions = np.linspace(right_start, right_end, 4)
    
    # Draw the bars and error bars on the subplots
    for i, function_id in enumerate(function_ids):
        # Get the mean and std values for single and mixed workloads
        if len(latencies_single[function_id]) > 0:
            single_mean = np.mean(latencies_single[function_id])
            single_std = np.std(latencies_single[function_id])
        else:
            single_mean = 0
            single_std = 0
            
        if len(latencies_mixed[function_id]) > 0:
            mixed_mean = np.mean(latencies_mixed[function_id])
            mixed_std = np.std(latencies_mixed[function_id])
        else:
            mixed_mean = 0
            mixed_std = 0
        
        # Draw bars in bottom subplot (all workloads)
        # For non-CNN bars, show the full height
        if function_id != 2:  # Top-K, Logit, Norm
            # Plot left part (Single workloads) - full height
            if single_mean > 0:
                ax_bottom.bar(
                    left_positions[i], 
                    single_mean, 
                    width=bar_width, 
                    color=function_colors[i],
                    #hatch=function_hatches[i],
                    yerr=single_std,
                    capsize=5,
                    label=function_names[i],
                    edgecolor='black',  # Add black border
                    linewidth=1.5       # Make border thicker
                )
            
            # Plot right part (Mixed workloads) - full height
            if mixed_mean > 0:
                ax_bottom.bar(
                    right_positions[i], 
                    mixed_mean, 
                    width=bar_width, 
                    color=function_colors[i],
                    #hatch=function_hatches[i],
                    yerr=mixed_std,
                    capsize=5,
                    edgecolor='black',  # Add black border
                    linewidth=1.5       # Make border thicker
                )
        else:  # CNN
            # For CNN, show truncated bars in bottom subplot
            # Plot left part (Single CNN) - truncated to low_max
            if single_mean > 0:
                # Plot truncated bar in bottom subplot
                ax_bottom.bar(
                    left_positions[i], 
                    min(single_mean, low_max), 
                    width=bar_width, 
                    color=function_colors[i],
                    #hatch=function_hatches[i],
                    capsize=0,  # No error bars for truncated bars
                    label=function_names[i],
                    edgecolor='black',
                    linewidth=1.5
                )
                
                # Plot full bar in top subplot if it extends above high_min
                if single_mean > high_min:
                    ax_top.bar(
                        left_positions[i], 
                        min(single_mean, high_max) - high_min, 
                        width=bar_width, 
                        color=function_colors[i],
                        #hatch=function_hatches[i],
                        yerr=single_std if single_mean < high_max else 0,
                        capsize=5,
                        bottom=high_min,
                        edgecolor='black',
                        linewidth=1.5
                    )
            
            # Plot right part (Mixed CNN) - truncated to low_max
            if mixed_mean > 0:
                # Plot truncated bar in bottom subplot
                ax_bottom.bar(
                    right_positions[i], 
                    min(mixed_mean, low_max), 
                    width=bar_width, 
                    color=function_colors[i],
                    #hatch=function_hatches[i],
                    capsize=0,  # No error bars for truncated bars
                    edgecolor='black',
                    linewidth=1.5
                )
                
                # Plot full bar in top subplot if it extends above high_min
                if mixed_mean > high_min:
                    ax_top.bar(
                        right_positions[i], 
                        min(mixed_mean, high_max) - high_min, 
                        width=bar_width, 
                        color=function_colors[i],
                       #hatch=function_hatches[i],
                        yerr=mixed_std if mixed_mean < high_max else 0,
                        capsize=5,
                        bottom=high_min,
                        edgecolor='black',
                        linewidth=1.5
                    )

    # Draw a vertical line to separate single from mixed workloads on both subplots
    separator = (left_positions[-1] + right_positions[0]) / 2
    ax_top.axvline(x=separator, color='black', linestyle='--', linewidth=2)
    ax_bottom.axvline(x=separator, color='black', linestyle='--', linewidth=2)
    
    # Set the same x limits for both subplots
    x_min = min(left_positions) - bar_width
    x_max = max(right_positions) + bar_width
    ax_top.set_xlim(x_min, x_max)
    ax_bottom.set_xlim(x_min, x_max)
    
    # Add cut-out marks to show the broken y-axis
    d = 0.015  # Size of diagonal lines in axes coordinates
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # Bottom-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Bottom-right diagonal
    
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Top-left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Top-right diagonal
    
    
    # Set x-axis ticks and labels only on the bottom subplot
    all_positions = np.concatenate([left_positions, right_positions])
    
    # Create labels for all positions, using function names for left part and empty strings for right part
    all_labels = list(function_names) + [""] * 4
    
    # Add "Mixed Workload" as a separate text label in the exact middle of the right positions
    # Calculate a position that's a bit more to the right than the center
    right_center = np.mean(right_positions)
    right_width = right_positions[-1] - right_positions[0]
    mixed_label_pos = right_center + right_width * 0.08  # Shift more to the right
    
    # Gap between the x-axis line and its tick labels (points)
    xtick_pad = 16

    # Position it at the same vertical level as the other tick labels, offset in points
    ax_bottom.annotate('Mixed Workload', xy=(mixed_label_pos, 0),
                       xycoords=ax_bottom.get_xaxis_transform(),
                       xytext=(0, -(xtick_pad + 2)), textcoords='offset points',
                       horizontalalignment='center', verticalalignment='top',
                       fontsize=18, fontweight='medium')
    
    # Set the ticks and labels
    ax_bottom.set_xticks(all_positions)
    ax_bottom.set_xticklabels(all_labels, fontsize=20, fontweight='medium')
    ax_bottom.tick_params(axis='x', pad=xtick_pad)
    ax_top.set_xticks([])  # No x ticks on top subplot
    
    # Format y-axis ticks
    ax_top.yaxis.set_major_formatter(ScalarFormatter())
    ax_bottom.yaxis.set_major_formatter(ScalarFormatter())
    
    # Set y-axis tick label font size to 17
    ax_top.tick_params(axis='y', labelsize=20)
    ax_bottom.tick_params(axis='y', labelsize=20)
    for tick in ax_top.get_yticklabels():
        tick.set_fontweight('medium')
    for tick in ax_bottom.get_yticklabels():
        tick.set_fontweight('medium')
    
    # Set y-axis label only on bottom subplot, positioned to cover both subplots
    fig.text(
        0.0, 0.5, 'Latency (μs)', va='center', rotation='vertical', fontsize=20, fontweight='medium'
    )
    
    # Add more space at top for legend and bottom for x-axis labels, without changing total figure size
    plt.subplots_adjust(left=0.16, top=0.9, bottom=0.22)
    
    # Add legend in one column on the left side, vertically centered on the axis break
    # so it spans across the cut between the two subplots
    gap_center_y = (ax_top.get_position().y0 + ax_bottom.get_position().y1) / 2
    axes_left_x = ax_bottom.get_position().x0
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=function_colors[i], 
                       #hatch=function_hatches[i], 
                      edgecolor='black', linewidth=1.5) for i in range(len(function_legend_names))],
        function_legend_names,
        loc='center left',
        bbox_to_anchor=(axes_left_x + 0.01, gap_center_y + 0.075),  # Just inside the left edge, shifted up
        ncol=1,        # One column of four entries
        frameon=True,
        prop={'size': 17, 'weight': 'medium'},
        labelspacing=0.6,  # Vertical gap between legend rows
        framealpha=0.5,
        columnspacing=0.8,  # Space between columns
        handletextpad=0.4  # Space between marker and text
    )
    
    # Remove top subplot's bottom and bottom subplot's top spines
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    
    # Save plot
    plt.savefig('mixed_workload_fig_14.pdf', bbox_inches='tight')
    print("\nPlot saved as mixed_workload_fig_14.pdf")

if __name__ == "__main__":
    create_mixed_figure()