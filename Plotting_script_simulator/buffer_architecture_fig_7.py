import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plot_style

plot_style.apply()
# Printed at \columnwidth = 240 pt in the paper; its tight-cropped PDF is ~1111 pt wide.
pstyle = plot_style.paper_style(printed_width_pt=240.0, cropped_width_pt=1111.1)

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
        1: 0.1,  # Multi- distribution of request size
        2: 0.3,
        4: 0.3,
        8: 0.3
    },
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
    "Single-fragment reqs \n dominant",
    "Multi-fragment reqs \n dominant (Small scale)",
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

# Function definitions remain the same as the provided code


def plot_three_figures(data_list_1, data_list_2, output_dir, request_lengths, request_lengths_large):
    architectures = ["linear/8/three_type_equal/PER_CORE_QUEUE/RR/RR", 
                     "linear/8/three_type_equal/REASSEMBLY_BUFFER/RR/RR"]
    bar_width = 0.35
    num_request_lengths = len(request_lengths) + 1
    num_request_lengths_large = len(request_lengths_large) + 1

    x_positions_1 = np.arange(num_request_lengths)
    x_positions_2 = np.arange(num_request_lengths_large)

    labels = ["Per-Acc Buffer", "Reassem. Buffer"]  # short, one line each
    styles = [
        {'color': plot_style.C_BASE, 'edgecolor': 'black', 'hatch': ''},   # per-accelerator buffer (baseline)
        {'color': plot_style.C_OURS, 'edgecolor': 'black', 'hatch': ''}    # reassembly buffer (ours)
    ]

    fig, axs = plt.subplots(1, 3, figsize=(18, 3.7))  # single row; 3.7 in keeps the printed height at ~74 pt

    for i, (data_list, request_lengths, x_positions, col_idx, y_label) in enumerate(
            [(data_list_1[0], request_lengths, x_positions_1, 0, "Request Fail Rate (%)"), 
             (data_list_1[1], request_lengths, x_positions_1, 1, ""), 
             (data_list_2[1], request_lengths_large, x_positions_2, 2, "")]):
        
        ax = axs[col_idx]
        for arch_idx, arch in enumerate(architectures):
            df_list = data_list[arch]
            drop_rate_means, drop_rate_stds = calculate_mean_std(df_list, request_lengths)
            style = styles[arch_idx]
            ax.bar(x_positions + arch_idx * bar_width, drop_rate_means, bar_width,
                   yerr=drop_rate_stds, error_kw=pstyle.error_kw,
                   color=style['color'], edgecolor=style['edgecolor'], hatch=style['hatch'], linewidth=pstyle.edge)
        
        # Set titles and labels
        ax.set_title(
            DISTRIBUTION_LABELS[col_idx] if col_idx < 2 else "Multi-fragment reqs \n dominant (Large Scale)",
            fontsize=pstyle.font,
            fontweight='medium',
        )
        if col_idx == 0:
            ax.set_ylabel(y_label, fontsize=pstyle.font, fontweight='medium')
        ax.set_xticks(x_positions + bar_width / 2)  # centre of each pair of bars
        ax.set_xticklabels([str(rl) for rl in request_lengths] + ['Total'], fontsize=pstyle.font, fontweight='medium')
        
        
        # Set y-axis limits
        # if col_idx < 2:
        #     ax.set_ylim(0, 70)
        # else:
        ax.set_ylim(0, 100)

        if col_idx == 0:
            ax.set_ylabel(y_label, fontsize=pstyle.font, fontweight='medium')
        else:
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])  # Disable tick labels for other subplots
        
        if col_idx == 0:
            legend = ax.legend(labels, loc="upper left", ncol=1, prop={'weight': 'medium', 'size': pstyle.font})
            ax.tick_params(axis='y', labelsize=pstyle.font)

        for label in ax.get_yticklabels():
            label.set_fontweight('medium')

    # Add a single x-axis label centered below all subplots
    # Shared x label: centred under the three panels (not under the figure, whose centre is
    # shifted by the y-axis label) and just below the lowest tick label, both measured after
    # the final layout so they follow the font size.
    fig.subplots_adjust(wspace=0.02)  # horizontal spacing between subplots
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lowest = min(ax.get_tightbbox(renderer).y0 for ax in axs)   # display units
    y_label = fig.transFigure.inverted().transform((0, lowest))[1] - 0.01
    x_label = (axs[0].get_position().x0 + axs[-1].get_position().x1) / 2
    fig.text(x_label, y_label, "Request Size (Fragments)", ha='center', va='top',
             fontsize=pstyle.font, fontweight='medium')

    # Adjust layout and save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'buffer_architecture_fig_7.pdf'), bbox_inches='tight')
    pstyle.report(os.path.join(output_dir, 'buffer_architecture_fig_7.pdf'))

# Replace the call to `plot_four_figures` with `plot_three_figures` in the main function
def main():
    base_dir = os.path.join(DATA_DIR, "Components")
    base_dir_2 = os.path.join(DATA_DIR, "Large_ARCH")
    subdirs = ["linear/8/three_type_equal/PER_CORE_QUEUE/RR/RR", 
               "linear/8/three_type_equal/REASSEMBLY_BUFFER/RR/RR"]

    runs = [f"{i}" for i in range(1, 11)]
    output_dir = PLOT_DIR
    request_lengths = [1, 2, 4, 8]
    request_lengths_large = [1, 4, 64, 128]

    # Read data for the first two figures
    data_list_1 = []
    for packet_distribution in PACKET_DISTRIBUTIONS_1:
        data = read_csv_data(base_dir, subdirs, packet_distribution, runs)
        data_list_1.append(data)

    # Read data for the large-scale figure
    data_list_2 = []
    for packet_distribution in PACKET_DISTRIBUTIONS_2:
        data = read_csv_data(base_dir_2, subdirs, packet_distribution, runs)
        data_list_2.append(data)

    # Calculate and print numerical differences
    print("=== NUMERICAL RESULTS: RATE DIFFERENCES ===")
    print("Format: [Request Length 1, 2, 4, 8, Total] or [Request Length 1, 4, 64, 128, Total]")
    print("Per-Accelerator Buffer vs Reassembly Buffer")
    print()
    
    # Subplot 1: Single-fragment reqs dominant (Small Scale)
    print("SUBPLOT 1 - Single-fragment reqs dominant (Small Scale):")
    arch1_data_s1 = data_list_1[0][subdirs[0]]  # Per-Accelerator Buffer
    arch2_data_s1 = data_list_1[0][subdirs[1]]  # Reassembly Buffer
    
    means1_s1, stds1_s1 = calculate_mean_std(arch1_data_s1, request_lengths)
    means2_s1, stds2_s1 = calculate_mean_std(arch2_data_s1, request_lengths)
    
    print(f"Per-Accelerator Buffer rates (%): {[f'{m:.2f}' for m in means1_s1]}")
    print(f"Reassembly Buffer rates (%): {[f'{m:.2f}' for m in means2_s1]}")
    differences_s1 = [m2 - m1 for m1, m2 in zip(means1_s1, means2_s1)]
    print(f"Differences (RB - PAB) (%): {[f'{d:.2f}' for d in differences_s1]}")
    print()
    
    # Subplot 2: Multi-fragment reqs dominant (Small Scale)
    print("SUBPLOT 2 - Multi-fragment reqs dominant (Small Scale):")
    arch1_data_s2 = data_list_1[1][subdirs[0]]  # Per-Accelerator Buffer
    arch2_data_s2 = data_list_1[1][subdirs[1]]  # Reassembly Buffer
    
    means1_s2, stds1_s2 = calculate_mean_std(arch1_data_s2, request_lengths)
    means2_s2, stds2_s2 = calculate_mean_std(arch2_data_s2, request_lengths)
    
    print(f"Per-Accelerator Buffer rates (%): {[f'{m:.2f}' for m in means1_s2]}")
    print(f"Reassembly Buffer rates (%): {[f'{m:.2f}' for m in means2_s2]}")
    differences_s2 = [m2 - m1 for m1, m2 in zip(means1_s2, means2_s2)]
    print(f"Differences (RB - PAB) (%): {[f'{d:.2f}' for d in differences_s2]}")
    print()
    
    # Subplot 3: Multi-fragment reqs dominant (Large Scale)
    print("SUBPLOT 3 - Multi-fragment reqs dominant (Large Scale):")
    arch1_data_s3 = data_list_2[1][subdirs[0]]  # Per-Accelerator Buffer
    arch2_data_s3 = data_list_2[1][subdirs[1]]  # Reassembly Buffer
    
    means1_s3, stds1_s3 = calculate_mean_std(arch1_data_s3, request_lengths_large)
    means2_s3, stds2_s3 = calculate_mean_std(arch2_data_s3, request_lengths_large)
    
    print(f"Per-Accelerator Buffer rates (%): {[f'{m:.2f}' for m in means1_s3]}")
    print(f"Reassembly Buffer rates (%): {[f'{m:.2f}' for m in means2_s3]}")
    differences_s3 = [m2 - m1 for m1, m2 in zip(means1_s3, means2_s3)]
    print(f"Differences (RB - PAB) (%): {[f'{d:.2f}' for d in differences_s3]}")
    print()
    
    print("=== SUMMARY ===")
    print("Positive differences indicate Reassembly Buffer has higher fail rates")
    print("Negative differences indicate Reassembly Buffer has lower fail rates")
    print("PAB = Per-Accelerator Buffer, RB = Reassembly Buffer")
    print()

    # Plot three figures
    plot_three_figures(data_list_1, data_list_2, output_dir, request_lengths, request_lengths_large)

if __name__ == "__main__":
    main()