import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from matplotlib.lines import Line2D

import plot_fonts  # Helvetica Neue from fonts/, shared palette and print-size style

# Printed at 0.48\textwidth = 241.9 pt in the paper; its tight-cropped PDF is ~568 pt wide.
pstyle = plot_fonts.paper_style(printed_width_pt=241.9, cropped_width_pt=567.5)

class RawLatencyProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.thread_count = 0
        self.msg_size = 0
        self.function = 0
        self.median_latency = 0
        self.percentile_99 = 0
        self.raw_latencies_us = []
        
        # Extract details from directory path
        match_thread = re.search(r'n_(\d+)', directory_path)
        match_msg_size = re.search(r'm_(\d+)', directory_path)
        match_function = re.search(r'f_(\d+)', directory_path)
        
        self.thread_count = int(match_thread.group(1)) if match_thread else 0
        self.msg_size = int(match_msg_size.group(1)) if match_msg_size else 0
        self.function = int(match_function.group(1)) if match_function else 0

    def process_raw_latency_data(self):
        """Process raw latency data from all .txt files in the directory"""
        # Find all .txt files in the directory
        txt_files = glob.glob(os.path.join(self.directory_path, "*.txt"))
        
        if not txt_files:
            return False
        
        try:
            per_file = []
            
            # Process each .txt file: one latency sample (ns) per line
            for txt_file in txt_files:
                # Fast path: pandas' C parser (the files are plain integers,
                # up to ~2.5 M lines each). Blank lines become NaN so the line
                # count, and therefore the warm-up skip below, matches the file.
                try:
                    col = pd.read_csv(txt_file, header=None, usecols=[0], skip_blank_lines=False,
                                      dtype=np.float64, engine='c')[0].to_numpy()
                except pd.errors.EmptyDataError:
                    continue
                except ValueError:
                    # Non-numeric lines present: parse tolerantly and drop them
                    col = pd.to_numeric(
                        pd.read_csv(txt_file, header=None, usecols=[0], skip_blank_lines=False,
                                    dtype=str, engine='c', na_filter=False)[0],
                        errors='coerce').to_numpy(dtype=np.float64)
                
                # Skip the first 10% of the file (warm-up), then drop unparsable lines
                skip_count = len(col) // 10
                file_latencies_ns = col[skip_count:]
                file_latencies_ns = file_latencies_ns[~np.isnan(file_latencies_ns)]
                
                per_file.append(file_latencies_ns)
            
            all_latencies_ns = np.concatenate(per_file) if per_file else np.array([])
            
            if len(all_latencies_ns) == 0:
                print(f"No valid latency data found in directory: {self.directory_path}")
                return False
            
            # Convert from nanoseconds to microseconds and store for external access
            self.raw_latencies_us = all_latencies_ns / 1000.0
            
            # Calculate median and 99th percentile
            self.median_latency = np.median(self.raw_latencies_us)
            self.percentile_99 = np.percentile(self.raw_latencies_us, 99)
            
            return True
            
        except Exception as e:
            print(f"Error processing files in {self.directory_path}: {e}")
            return False


def get_tail_latency_directories(base_directory, instance_count, msg_size=1024, machine_type=1):
    """Get directories containing raw tail latency data for O_1 (OffRAC), O_2 (CPU), or O_3 (DPU)"""
    directory = f"{base_directory}/top_k_{instance_count}_inst"
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    # Look for directories matching the pattern based on machine type
    if machine_type == 1:  # OffRAC (O_1)
        dir_pattern = f"rr_d_*_m_{msg_size}_n_*_f_1_O_1"
    elif machine_type == 2:  # CPU (O_2)
        dir_pattern = f"rr_d_*_m_{msg_size}_n_*_f_1_O_2"
    elif machine_type == 3:  # DPU (O_3)
        dir_pattern = f"rr_d_*_m_{msg_size}_n_*_f_1_O_3"
    else:
        return []
    
    matching_dirs = glob.glob(os.path.join(directory, dir_pattern))
    
    return sorted(matching_dirs)


def collect_latency_data_for_clients(base_directory, instance_count, msg_size, machine_type, target_clients):
    """Collect raw latency data for specific client count"""
    directories = get_tail_latency_directories(base_directory, instance_count, msg_size, machine_type)
    
    all_latencies = []
    
    # Calculate the thread count we need to look for based on machine type
    if machine_type == 1:  # OffRAC - thread_count * instance_count = target_clients
        target_thread_count = target_clients // instance_count
    else:  # CPU and DPU - thread_count = target_clients (no multiplication)
        target_thread_count = target_clients
    
    print(f"Looking for directories with n_{target_thread_count} (machine_type={machine_type}, instance_count={instance_count})")
    
    for dir_path in directories:
        processor = RawLatencyProcessor(dir_path)
        
        # Check if this directory matches our target thread count
        if processor.thread_count == target_thread_count:
            if processor.process_raw_latency_data():
                all_latencies.append(processor.raw_latencies_us)
                print(f"Processed {dir_path}: {len(processor.raw_latencies_us)} latency points")
    
    return np.concatenate(all_latencies) if all_latencies else np.array([])


def calculate_cdf(data):
    """Calculate CDF for given data"""
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y


def create_cdf_comparison_plot(base_directory="data/scalability"):
    """Latency CDF of fRAC, CPU and DPU with 1, 2 and 4 instances each, on a broken x-axis.

    The left panel zooms on 0-15 us, where all three fRAC curves rise (8-14 us); the right
    panel shows 100-650 us, where the CPU and DPU curves rise. Only the low tails of the
    CPU/DPU curves (below CDF 0.03) fall in the hidden 15-100 us gap. Panels and cut marks
    follow azure_trace_fig_18.py so the two figures match side by side in the paper.
    """

    msg_size = 4096
    target_clients = 28

    # Define configurations to compare
    configs = [
        (1, 1, "1 Accel"),
        (2, 1, "2 Accel"),
        (4, 1, "4 Accel"),
        (1, 2, "CPU 1c"),
        (2, 2, "CPU 2c"),
        (4, 2, "CPU 4c"),
        (1, 3, "DPU 1c"),
        (2, 3, "DPU 2c"),
        (4, 3, "DPU 4c")
    ]

    # Color-blind friendly palette
    colors = [plot_fonts.C_OURS, plot_fonts.C_BASE, plot_fonts.C_THIRD] * 3  # fRAC, CPU, DPU for each instance count
    line_styles = ['-', '-', '-', '--', '--', '--', ':', ':', ':']  # Solid for Accel, Dashed for CPU, Dotted for DPU

    # Two panels sharing the y-axis. The right panel gets 2.5x the width: it carries six
    # curves over 550 us plus the nine-entry legend (~98 pt printed, which has to fit in
    # the 180-575 us band, see below), the left one three curves over 15 us.
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, figsize=(8, 2.9),
                                            gridspec_kw={'width_ratios': [1, 2.5]})

    all_data = {}   # label -> sorted latencies (us)
    handles = []

    # Collect data for all configurations and draw each curve on both panels
    for i, (instance_count, machine_type, label) in enumerate(configs):
        print(f"Collecting data for {label} with {msg_size}B and {target_clients} clients...")
        latencies = collect_latency_data_for_clients(base_directory, instance_count, msg_size, machine_type, target_clients)

        if len(latencies) > 0:
            print(f"{label}: {len(latencies)} latency points")
            x, y = calculate_cdf(latencies)
            all_data[label] = x
            ax_left.plot(x, y, color=colors[i], linewidth=pstyle.curve, linestyle=line_styles[i])
            ax_right.plot(x, y, color=colors[i], linewidth=pstyle.curve, linestyle=line_styles[i])
            handles.append(Line2D([0], [0], color=colors[i], linestyle=line_styles[i], lw=pstyle.curve, label=label))
        else:
            print(f"No data found for {label}!")

    # Broken x-axis: left 0-15 us (fRAC), right 100-650 us (CPU, DPU)
    ax_left.set_xlim(0, 15)
    ax_left.set_xticks(np.arange(0, 16, 5))
    ax_right.set_xlim(100, 650)
    # No label at the right panel's left edge: "100" there would collide with "15" across
    # the cut and tight_layout would double the gap between the panels (fig 18 likewise
    # leaves its right panel's edge unlabelled).
    ax_right.set_xticks(np.arange(200, 651, 100))
    ax_left.set_ylim(0, 1)

    # Hide the spines between the panels
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

    # Labels, ticks and grid (one centred x label for both panels, as in fig 18)
    fig.supxlabel('Latency (μs)', fontsize=pstyle.font, y=0.08, fontweight='medium')
    ax_left.set_ylabel('CDF', fontsize=pstyle.font, fontweight='medium')
    for ax in (ax_left, ax_right):
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        ax.tick_params(axis='both', which='major', labelsize=pstyle.font)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('medium')

    # Legend: one column per system (Accel / CPU / DPU), in the upper band of the right
    # panel (CDF ~0.52-0.97). In that band the panel is empty between the CPU 4c rise
    # (~175 us) and the DPU / CPU 1c rises (~570 us) except for the vertical step of
    # CPU 2c at ~385 us, which the half-transparent frame lets show through; this is the
    # same spot the legend occupied on the single-axis version. The script reports after
    # layout which curves the legend box touches.
    legend = ax_right.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(378, 0.975), bbox_transform=ax_right.transData,  # centre of the 180-575 us band
        ncol=3,
        prop={'size': pstyle.font, 'weight': 'medium'},
        labelspacing=0.3,
        columnspacing=0.4,
        handlelength=1.2,
        handletextpad=0.3,
        borderpad=0.3,
        borderaxespad=0.0,
        framealpha=0.5,
    )

    plt.tight_layout()

    # Report which curves pass through the legend box
    fig.canvas.draw()
    # matplotlib < 3.5 needs the renderer passed explicitly
    bb = legend.get_window_extent(fig.canvas.get_renderer())
    inv = ax_right.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    pos_l, pos_r = ax_left.get_position(), ax_right.get_position()
    fw = fig.get_size_inches()[0] * 72 * pstyle.scale   # figure width in printed pt
    print(f"Panels (printed pt): left {pos_l.width * fw:.0f}, gap {(pos_r.x0 - pos_l.x1) * fw:.0f}, right {pos_r.width * fw:.0f}")
    print(f"Legend box: {x0:.0f}-{x1:.0f} μs x CDF {y0:.2f}-{y1:.2f} (right panel)")
    for label, x in all_data.items():
        lo, hi = np.quantile(x, [y0, y1])   # where the curve is while inside the legend's CDF band
        if hi >= x0 and lo <= x1:
            print(f"  legend covers {label} between {max(lo, x0):.0f} and {min(hi, x1):.0f} μs")

    # Print statistics for all configurations
    for label, latencies in all_data.items():
        print(f"\n{label} Statistics:")
        print(f"  Median: {np.median(latencies):.2f} μs")
        print(f"  90th percentile: {np.percentile(latencies, 90):.2f} μs")
        print(f"  95th percentile: {np.percentile(latencies, 95):.2f} μs")
        print(f"  99th percentile: {np.percentile(latencies, 99):.2f} μs")

    plt.savefig('tail_latency_cdf_fig_17.pdf', bbox_inches='tight', dpi=300)
    pstyle.report('tail_latency_cdf_fig_17.pdf')
    print(f"\nPlot saved as tail_latency_cdf_fig_17.pdf")


if __name__ == "__main__":
    create_cdf_comparison_plot("data/scalability")
