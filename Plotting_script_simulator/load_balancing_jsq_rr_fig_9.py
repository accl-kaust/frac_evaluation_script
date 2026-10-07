import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

import plot_style
from plot_colors import (
    BAR_PALETTE, BAR_INDICES, BAR_SATURATION, BAR_WHITE, BAR_BLACK, toned_palette,
)

# Req A/B use Figure 5's exact green/purple; Req C is a lighter yellow.
request_colors = [BAR_INDICES[1], BAR_INDICES[0], 8]  # Req A, B, C
white = [BAR_WHITE[1], BAR_WHITE[0], 0.25]
black = [BAR_BLACK[1], BAR_BLACK[0], 0.0]
request_palette = [
    toned_palette([index], palette=BAR_PALETTE, saturation=BAR_SATURATION,
                  white=white[n], black=black[n])[0]
    for n, index in enumerate(request_colors)
]

plot_style.apply()
# Printed at \columnwidth = 240 pt in the paper; its tight-cropped PDF is ~696 pt wide.
pstyle = plot_style.paper_style(printed_width_pt=240.0, cropped_width_pt=696.4)

# Inputs live in <repo>/Data_simulation and figures go to <repo>/Plot, both resolved
# relative to this file so the script works from any working directory.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_DIR, 'Data_simulation')
PLOT_DIR = os.path.join(_REPO_DIR, 'Plot')

PACKET_DISTRIBUTIONS = [
    {
        1: 0.1,  # 10% chance of 1 packet
        2: 0.3,  # 30% chance of 2 packets
        4: 0.3,  # 30% chance of 4 packets
        8: 0.3   # 30% chance of 8 packets
    },
    {
        1: 0.25,  # 25% chance of 1 packet
        2: 0.25,  # 25% chance of 2 packets
        4: 0.25,  # 25% chance of 4 packets
        8: 0.25   # 25% chance of 8 packets
    },
    {
        1: 0.95,  # 95% chance of 1 packet
        2: 0.03,  # 3% chance of 2 packets
        4: 0.01,  # 1% chance of 4 packets
        8: 0.01   # 1% chance of 8 packets
    }
]

label_to_folder = {
    "Multi-packet req dominant": PACKET_DISTRIBUTIONS[0],
    "Even distribution of req sizes": PACKET_DISTRIBUTIONS[1],
    "Single packet req dominant": PACKET_DISTRIBUTIONS[2]
}

def find_csv_file(folder, filename='latency.csv'):
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        return file_path

    for root, _, files in os.walk(folder):
        if filename in files:
            return os.path.join(root, filename)
    return None

def read_csv_data(base_dir, packet_distribution):
    data = {}
    distribution_str = str(packet_distribution)
    
    subdirs = [
        f"JSQ/Inter_Packet_5/{distribution_str}",
        f"RR/Inter_Packet_5/{distribution_str}"
    ]
    
    for subdir in subdirs:
        folder = os.path.join(base_dir, subdir)
        file_path = find_csv_file(folder)
        if file_path:
            df = pd.read_csv(file_path)
            data[subdir] = df
        else:
            print(f"File not found in {folder} or its subdirectories")
    return data

def separate_workload_types(df):
    workload_a = df[df['Workload Type'] == 'A']['Core Queueing Time']
    workload_b = df[df['Workload Type'] == 'B']['Core Queueing Time']
    workload_c = df[df['Workload Type'] == 'C']['Core Queueing Time']
    return workload_a, workload_b, workload_c

def plot_latency(all_data, output_dir):
    distributions = ["Single packet req dominant", "Even distribution of req sizes", "Multi-packet req dominant"]
    workload_types = ['A', 'B', 'C']  # Keep single letters for data processing
    workload_labels = ['Req A', 'Req B', 'Req C']  # Full labels for legend
    colors = request_palette  # Req A, B, C
    x_values = np.arange(len(distributions))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.16), sharey=True)  # 3.16 in keeps the printed height at ~70 pt
    plt.subplots_adjust(top=0.9, wspace=0.0)  # No space between subplots

    for ax, scheduling in zip(axes, ["JSQ", "RR"]):
        for idx, workload in enumerate(workload_types):
            workload_data = []

            for distribution in distributions:
                key = f"{scheduling}/Inter_Packet_5/{str(label_to_folder[distribution])}"
                if key in all_data:
                    df = all_data[key]
                    workload_a, workload_b, workload_c = separate_workload_types(df)
                    if workload == 'A':
                        workload_data.append(workload_a)
                    elif workload == 'B':
                        workload_data.append(workload_b)
                    elif workload == 'C':
                        workload_data.append(workload_c)

            # Plot boxplots with narrower width
            spacing_offsets = [-0.25, 0, 0.25]
            ax.boxplot(
                workload_data,
                positions=x_values + spacing_offsets[idx],
                widths=0.2,
                patch_artist=True,
                boxprops=dict(facecolor=colors[idx], color='black', linewidth=pstyle.edge),
                whiskerprops=dict(color='black', linewidth=pstyle.edge),
                medianprops=dict(color='black' if idx == 0 else colors[idx], linewidth=pstyle.median),
                showfliers=False
            )
        
        # Configure the subplot
        ax.set_xticks(x_values)
        ax.set_xticklabels(["S", "E", "M"], fontsize=pstyle.font, fontweight='medium')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # four evenly spaced y ticks, as in the paper
        ax.set_title(f"{scheduling}", fontsize=pstyle.font, fontweight='medium')
        ax.tick_params(axis='both', labelsize=pstyle.font)
        for label in ax.get_yticklabels():
            label.set_fontweight('medium')

        if ax == axes[0]:
            # Left subplot: Add ylabel and legend
            ax.set_ylabel('Queueing Latency (μs)', fontsize=pstyle.font, fontweight='medium')
            handles = [mpatches.Patch(color=colors[i], label=workload_labels[i]) for i in range(len(workload_types))]
            legend = ax.legend(handles=handles, loc='upper left', ncol=1, prop={'weight': 'medium', 'size': pstyle.font})
            # Make legend text medium
        else:
            # Right subplot: No ylabel
            ax.tick_params(axis='y', left=False, labelleft=False)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'load_balancing_jsq_rr_fig_9.pdf'), bbox_inches='tight')
    pstyle.report(os.path.join(output_dir, 'load_balancing_jsq_rr_fig_9.pdf'))
    plt.close()

def main():
    base_dir = os.path.join(DATA_DIR, "LOAD_BALANCE/1/linear/8/three_type_95/REASSEMBLY_BUFFER/Prioritized_RR")
    output_dir = PLOT_DIR
    all_data = {}

    # Load data for all distributions
    for label, folder in label_to_folder.items():
        all_data.update(read_csv_data(base_dir, folder))
    
    plot_latency(all_data, output_dir)

if __name__ == "__main__":
    main()