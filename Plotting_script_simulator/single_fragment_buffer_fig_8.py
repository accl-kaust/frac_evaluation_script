import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from plot_colors import bar_palette

# Figure 5's purple/green; the shared color controls live in plot_colors.py.
color_baseline, color_improved = bar_palette()

plot_style.apply()
# Printed at \columnwidth = 240 pt in the paper; its tight-cropped PDF is ~696 pt wide.
pstyle = plot_style.paper_style(printed_width_pt=240.0, cropped_width_pt=696.3)

# Inputs live in <repo>/Data_simulation and figures go to <repo>/Plot, both resolved
# relative to this file so the script works from any working directory.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_DIR, 'Data_simulation')
PLOT_DIR = os.path.join(_REPO_DIR, 'Plot')

INTER_PACKET_INTERVAL = "5.0"  # Replace with the actual value
PACKET_DISTRIBUTIONS_1 = [
    {
        1: 0.95,  # Single packet requests dominant
        2: 0.03,
        4: 0.01,
        8: 0.01
    },
    {
        1: 0.25,  # Multi-packet requests dominant
        2: 0.25,
        4: 0.25,
        8: 0.25
    }
]

PACKET_DISTRIBUTIONS_2 = [
    {
        1: 0.95,  # Single packet requests dominant
        4: 0.03,
        64: 0.01,
        128: 0.01
    },
    {
        1: 0.1,  # Multi- distribution of request size
        4: 0.3,
        64: 0.3,
        128: 0.3
    },
]
DISTRIBUTION_LABELS = [
    "Single-fragment reqs dominant",
    "Even distribution of req size",
    "Multi-fragment reqs dominant"
]

# Function to check for the CSV file in the base directory first and, if not found, search in subfolders
def find_csv_file(folder, filename='drop_rates_per_request_length.csv'):
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        return file_path
    for root, dirs, files in os.walk(folder):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Function to read the CSV files from different run directories and return combined data
def read_csv_data(base_dir, subdirs, packet_distribution, runs):
    all_run_data = {subdir: [] for subdir in subdirs}

    for run in runs:
        for subdir in subdirs:
            folder = os.path.join(base_dir, run, subdir)
            folder = os.path.join(folder, "Inter_Packet_" + str(INTER_PACKET_INTERVAL))
            folder = os.path.join(folder, str(packet_distribution))
            file_path = find_csv_file(folder)
            if file_path:
                df = pd.read_csv(file_path)
                all_run_data[subdir].append(df)
            else:
                print(f"File not found in {folder} or its subdirectories")
    return all_run_data

# Function to calculate the mean and standard deviation for drop rates, including total
def calculate_mean_std(df_list, request_lengths):
    drop_rate_means = []
    drop_rate_stds = []
    total_drop_rates = []
    
    for request_length in request_lengths:
        drop_rates = []
        for df in df_list:
            sub_data = df[df['Request Length'] == request_length]
            avg_drop_rate = sub_data['Drop Rate'].mean() * 100  # Convert to percentage
            drop_rates.append(avg_drop_rate)

        drop_rate_means.append(np.mean(drop_rates))
        drop_rate_stds.append(np.std(drop_rates))

    # Calculate total drop rate for all runs
    for df in df_list:
        total_dropped_packets = df['Dropped Packets'].sum()
        total_sent_packets = df['Sent Packets'].sum()
        total_drop_rate = (total_dropped_packets / total_sent_packets * 100) if total_sent_packets > 0 else 0
        total_drop_rates.append(total_drop_rate)

    # Calculate mean and standard deviation for total drop rate
    drop_rate_means.append(np.mean(total_drop_rates))
    drop_rate_stds.append(np.std(total_drop_rates))

    return drop_rate_means, drop_rate_stds

def plot_two_subfigures(data1, data2, output_dir, request_lengths1, request_lengths2):
    architectures = [
        "linear/8/three_type_equal/REASSEMBLY_BUFFER/RR/RR",
        "linear/8/three_type_equal/REASSEMBLY_BUFFER/Prioritized_RR/RR"
    ]
    bar_width = 0.25  # Width of each individual bar (reduced)
    num_request_lengths1 = len(request_lengths1) + 1  # +1 for "Total" drop rate
    num_request_lengths2 = len(request_lengths2) + 1
    x_positions1 = np.arange(num_request_lengths1)  # Positions for 1, 2, 4, 8, Total
    x_positions2 = np.arange(num_request_lengths2)

    labels = ["Without Single-Frag. Buffer", "With Single-Frag. Buffer"]  # one line each
    styles = [
        {'color': color_baseline, 'edgecolor': 'black', 'hatch': ''},   # reassembly buffer alone (baseline)
        {'color': color_improved, 'edgecolor': 'black', 'hatch': ''},   # with single fragment buffer (ours)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.54), sharey=True)  # 3.54 in keeps the printed height at ~82 pt

    # Plot for base_dir
    for arch_idx, arch in enumerate(architectures):
        df_list = data1[arch]
        drop_rate_means, drop_rate_stds = calculate_mean_std(df_list, request_lengths1)

        style = styles[arch_idx]
        axes[0].bar(
            x_positions1 + arch_idx * bar_width, drop_rate_means, bar_width,
            yerr=drop_rate_stds, error_kw=pstyle.error_kw,
            color=style['color'], edgecolor=style['edgecolor'], hatch=style['hatch'], label=labels[arch_idx]
        , linewidth=pstyle.edge)

    axes[0].set_xticks(x_positions1 + bar_width / 2)  # centre of each pair of bars
    axes[0].set_xticklabels([str(rl) for rl in request_lengths1] + ['Total'], fontsize=pstyle.font, fontweight='medium')
    axes[0].set_title("Small Scale", fontsize=pstyle.font, fontweight='medium')
    axes[0].set_ylabel("Request Fail Rate (%)", fontsize=pstyle.font, fontweight='medium')
    axes[0].tick_params(axis='y', labelsize=pstyle.font)
    for label in axes[0].get_yticklabels():
        label.set_fontweight('medium')

    # Add legend in the left figure
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, legend_labels, loc='upper left', prop={'weight': 'medium', 'size': pstyle.font})

    # Plot for base_dir_2
    for arch_idx, arch in enumerate(architectures):
        df_list = data2[arch]
        drop_rate_means, drop_rate_stds = calculate_mean_std(df_list, request_lengths2)

        style = styles[arch_idx]
        axes[1].bar(
            x_positions2 + arch_idx * bar_width, drop_rate_means, bar_width,
            yerr=drop_rate_stds, error_kw=pstyle.error_kw,
            color=style['color'], edgecolor=style['edgecolor'], hatch=style['hatch'], linewidth=pstyle.edge
        )

    axes[1].set_xticks(x_positions2 + bar_width / 2)
    axes[1].set_xticklabels([str(rl) for rl in request_lengths2] + ['Total'], fontsize=pstyle.font, fontweight='medium')
    axes[1].set_title("Large Scale", fontsize=pstyle.font, fontweight='medium')

    # Disable y-axis on the right subplot
    axes[1].tick_params(axis='y', left=False, labelleft=False)


    # Adjust layout and save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.subplots_adjust(wspace=-1)  # Negative spacing to bring subfigures closer
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Apply tight layout for overall adjustments
    # Shared x label centred under the two panels (the figure centre is shifted by the y label).
    x_label = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    fig.text(x_label, 0.06, "Request Size (Fragments)", ha='center', fontsize=pstyle.font, fontweight='medium')
    plt.savefig(os.path.join(output_dir, 'single_fragment_buffer_fig_8.pdf'), bbox_inches='tight')
    pstyle.report(os.path.join(output_dir, 'single_fragment_buffer_fig_8.pdf'))
    #plt.show()

def main():
    base_dir = os.path.join(DATA_DIR, "Components")
    base_dir_2 = os.path.join(DATA_DIR, "Large_ARCH")
    subdirs = ["linear/8/three_type_equal/REASSEMBLY_BUFFER/RR/RR", 
               "linear/8/three_type_equal/REASSEMBLY_BUFFER/Prioritized_RR/RR"]
    runs = [f"{i}" for i in range(1, 11)]
    output_dir = PLOT_DIR
    request_lengths = [1, 2, 4, 8]
    request_lengths_large = [1, 4, 64, 128]

    data1 = read_csv_data(base_dir, subdirs, PACKET_DISTRIBUTIONS_1[-1], runs)
    data2 = read_csv_data(base_dir_2, subdirs, PACKET_DISTRIBUTIONS_2[-1], runs)

    # Calculate and print numerical differences
    print("=== NUMERICAL RESULTS: RATE DIFFERENCES ===")
    print("Format: [Request Length 1, 2, 4, 8, Total]")
    print("Reassembly Buffer vs Reassembly Buffer with Single Fragment Buffer")
    print()
    
    # Small Scale Analysis
    print("SMALL SCALE (Data1):")
    arch1_data = data1[subdirs[0]]  # Reassembly Buffer
    arch2_data = data1[subdirs[1]]  # Reassembly Buffer with Single Fragment Buffer
    
    means1, stds1 = calculate_mean_std(arch1_data, request_lengths)
    means2, stds2 = calculate_mean_std(arch2_data, request_lengths)
    
    print(f"Reassembly Buffer rates (%): {[f'{m:.2f}' for m in means1]}")
    print(f"Reassembly Buffer w/ SFB rates (%): {[f'{m:.2f}' for m in means2]}")
    differences1 = [m2 - m1 for m1, m2 in zip(means1, means2)]
    print(f"Differences (SFB - RB) (%): {[f'{d:.2f}' for d in differences1]}")
    print()
    
    # Large Scale Analysis  
    print("LARGE SCALE (Data2):")
    arch1_data_large = data2[subdirs[0]]  # Reassembly Buffer
    arch2_data_large = data2[subdirs[1]]  # Reassembly Buffer with Single Fragment Buffer
    
    means1_large, stds1_large = calculate_mean_std(arch1_data_large, request_lengths_large)
    means2_large, stds2_large = calculate_mean_std(arch2_data_large, request_lengths_large)
    
    print(f"Reassembly Buffer rates (%): {[f'{m:.2f}' for m in means1_large]}")
    print(f"Reassembly Buffer w/ SFB rates (%): {[f'{m:.2f}' for m in means2_large]}")
    differences2 = [m2 - m1 for m1, m2 in zip(means1_large, means2_large)]
    print(f"Differences (SFB - RB) (%): {[f'{d:.2f}' for d in differences2]}")
    print()
    
    print("=== SUMMARY ===")
    print("Positive differences indicate Reassembly Buffer with Single Fragment Buffer has higher fail rates")
    print("Negative differences indicate Reassembly Buffer with Single Fragment Buffer has lower fail rates")
    print()

    plot_two_subfigures(data1, data2, output_dir, request_lengths, request_lengths_large)

if __name__ == "__main__":
    main()