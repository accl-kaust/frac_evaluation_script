import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch

import plot_fonts  # noqa: F401  (Helvetica Neue from fonts/, medium weight everywhere)

# Define function names and colors
functions = [1, 3, 5]  # Top-K, Logit Transform, Normalization
function_names = ["Top-K", "Logit", "Norm"]
# Replace with color-blind friendly palette (Okabe-Ito color scheme)
colors = ['#0072B2', '#D55E00', '#009E73']  # Blue, Vermilion, Bluish green

# Target fragment sizes
target_fragments = [1, 2, 4]  # 1KB, 2KB, 4KB

class LogProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.avg_latency = 0
        
        # Extract details from filename
        file_name = os.path.basename(file_path)
        match_function = re.search(r'f_(\d+)', file_name)
        match_x_size = re.search(r'X_(\d+)', file_name)
        
        self.function = int(match_function.group(1)) if match_function else 0
        self.x_size = int(match_x_size.group(1)) if match_x_size else 0
        
        # Map X size to fragment count
        if self.x_size == 1024:
            self.fragment_count = 1
        elif self.x_size == 2048:
            self.fragment_count = 2
        elif self.x_size == 4096:
            self.fragment_count = 4
        else:
            self.fragment_count = self.x_size // 1024

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
                        if time >= 10:  # Filter data after 10 seconds
                            self.data.append(float(avg_latency))
                
                print(f"Extracted {len(self.data)} latency data points")
                return len(self.data) > 0
        except Exception as e:
            print(f"Error processing file {self.file_path}: {e}")
            return False


def collect_data_for_plot(directory):
    """Collect latency data for all functions and fragment sizes"""
    # Dictionary to store data: {function: {fragment_size: [latencies]}}
    plot_data = {func: {frag: [] for frag in target_fragments} for func in functions}
    
    # List all files in the directory
    files = os.listdir(directory)
    print(f"Found {len(files)} files in directory {directory}")
    
    # Process each file
    for file_name in files:
        if not file_name.endswith('.log'):
            continue
            
        file_path = os.path.join(directory, file_name)
        
        # Extract function and fragment size from filename
        match_function = re.search(r'f_(\d+)', file_name)
        match_x_size = re.search(r'X_(\d+)', file_name)
        
        if not match_function or not match_x_size:
            continue
            
        function = int(match_function.group(1))
        x_size = int(match_x_size.group(1))
        
        # Only process files for functions 1, 3, 5
        if function not in functions:
            continue
            
        # Map X size to fragment count
        if x_size == 1024:
            fragment_count = 1
        elif x_size == 2048:
            fragment_count = 2
        elif x_size == 4096:
            fragment_count = 4
        else:
            fragment_count = x_size // 1024
            
        # Only include fragment sizes 1, 2, 4
        if fragment_count not in target_fragments:
            continue
            
        # Process the log file
        processor = LogProcessor(file_path)
        if processor.parse_log():
            # Add the latency data to our plot_data dictionary
            plot_data[function][fragment_count].extend(processor.data)
    
    return plot_data


def create_grouped_bar_chart(data, directory):
    """Create a grouped bar chart comparing 'With' and 'Without' reassembly."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Colors for "With" and "Without" reassembly
    colors = {'with': '#0072B2', 'without': '#AACCFF'} # Light blue for 'without'


    n_functions = len(functions)
    n_fragments = len(target_fragments)
    bar_width = 0.35

    group_width = n_fragments * (bar_width * 3.5)

    all_tick_positions = []
    group_positions = []
    max_latency = 0

    for i, function in enumerate(functions):
        group_center = i * group_width
        group_positions.append(group_center)

        latencies_frag1 = data[function][1]
        if not latencies_frag1:
            print(f"Warning: No data for function {function_names[i]} with 1 fragment. Skipping 'Without Reassembly' calculation.")
            mean_frag1, std_frag1 = 0, 0
        else:
            mean_frag1 = np.mean(latencies_frag1)
            std_frag1 = np.std(latencies_frag1)

        for j, fragment in enumerate(target_fragments):
            pos = group_center + j * bar_width * 2.5

            # --- "With Reassembly" bar ---
            latencies_with = data[function][fragment]
            if latencies_with:
                mean_with = np.mean(latencies_with)
                std_with = np.std(latencies_with)
                ax.bar(pos - bar_width/2, mean_with, bar_width,
                       color=colors['with'], edgecolor='black', linewidth=1.5)
                ax.errorbar(pos - bar_width/2, mean_with, yerr=std_with,
                            fmt='none', ecolor='black', elinewidth=1.5, capsize=5)
                if mean_with + std_with > max_latency:
                    max_latency = mean_with + std_with

            # --- "Without Reassembly" bar ---
            if mean_frag1 > 0:
                mean_without = mean_frag1 * fragment
                std_without = std_frag1 * fragment
                ax.bar(pos + bar_width/2, mean_without, bar_width,
                       color=colors['without'], edgecolor='black', linewidth=1.5, hatch='//')
                ax.errorbar(pos + bar_width/2, mean_without, yerr=std_without,
                            fmt='none', ecolor='black', elinewidth=1.5, capsize=5)
                if mean_without + std_without > max_latency:
                    max_latency = mean_without + std_without
            
            all_tick_positions.append(pos)
    
    # Add vertical lines to separate function groups
    for i in range(n_functions - 1):
        # Calculate the midpoint between the last bar of one group and the first of the next
        last_bar_of_group = group_positions[i] + (n_fragments - 1) * bar_width * 2.5
        first_bar_of_next_group = group_positions[i+1]
        line_pos = (last_bar_of_group + first_bar_of_next_group) / 2
        ax.axvline(x=line_pos, color='black', linestyle='--', linewidth=2)

    # --- Customize plot ---
    ax.set_xticks(all_tick_positions)
    ax.set_xticklabels([str(f) for f in target_fragments] * n_functions, fontsize=20, fontweight='medium')
    
    # Add function names below fragment numbers
    for i, func_name in enumerate(function_names):
        group_center = group_positions[i]
        # Center the label within the group of bars
        label_pos = group_center + ( (n_fragments - 1) * bar_width * 2.5 ) / 2
        ax.text(
            label_pos,
            -0.13,
            func_name,
            ha='center',
            va='top',
            fontsize=20,
            fontweight='medium',
            transform=ax.get_xaxis_transform(),
        )

    # Set X and Y labels
    ax.set_xlabel("Fragments per Request", fontsize=20, fontweight='medium', labelpad=32)
    ax.set_ylabel("Latency(μs)", fontsize=20, fontweight='medium', y=0.5)
    ax.tick_params(axis='y', labelsize=20)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('medium')

    # Set y-axis limit
    ax.set_ylim(top=max_latency * 1.1 if max_latency > 0 else 25)

    # Add legend
    legend_patches = [
        Patch(facecolor=colors['with'], label='With Reassembly', edgecolor='black', linewidth=1.5),
        Patch(facecolor=colors['without'], label='Without Reassembly (emu)', edgecolor='black', linewidth=1.5, hatch='//')
    ]
    ax.legend(handles=legend_patches, loc='upper left', ncol=1, frameon=True, framealpha=0.7, prop={'size': 20, 'weight': 'medium'}, 
              bbox_to_anchor=(0.0, 1.00))
    
    plt.tight_layout(rect=[0, 0.18, 1, 1])

    plt.savefig("reassembly_fig_15.pdf", bbox_inches='tight')
    print("Plot saved as reassembly_fig_15.pdf")
    plt.close()


if __name__ == "__main__":
    directory = "data/reassembly"
    print(f"Looking for files in directory: {os.path.abspath(directory)}")
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        exit(1)
    
    # Collect data for all functions and fragment sizes
    plot_data = collect_data_for_plot(directory)
    
    # Create the grouped bar chart
    create_grouped_bar_chart(plot_data, directory)
    
    print("Plot completed!")