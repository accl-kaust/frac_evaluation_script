import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from matplotlib.ticker import ScalarFormatter

import plot_fonts  # noqa: F401  (Helvetica Neue from fonts/, medium weight everywhere)

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
            all_latencies_ns = []
            
            # Process each .txt file
            for txt_file in txt_files:
                # First pass: count total lines
                total_lines = 0
                with open(txt_file, 'r') as f:
                    for line in f:
                        total_lines += 1
                
                # Determine skip count: skip first 10% of file
                skip_count = total_lines // 10
                
                # Second pass: process the data
                file_latencies_ns = []
                with open(txt_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i < skip_count:
                            continue
                        
                        try:
                            latency_ns = float(line.strip())
                            file_latencies_ns.append(latency_ns)
                        except ValueError:
                            continue
                
                all_latencies_ns.extend(file_latencies_ns)
            
            if len(all_latencies_ns) == 0:
                print(f"No valid latency data found in directory: {self.directory_path}")
                return False
            
            # Convert from nanoseconds to microseconds and store for external access
            self.raw_latencies_us = [lat / 1000.0 for lat in all_latencies_ns]
            
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
                all_latencies.extend(processor.raw_latencies_us)
                print(f"Processed {dir_path}: {len(processor.raw_latencies_us)} latency points")
    
    return np.array(all_latencies)


def calculate_cdf(data):
    """Calculate CDF for given data"""
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y


def create_cdf_comparison_plot():
    """Create CDF comparison plot for all accelerator and CPU configurations"""
    
    base_directory = "data/scalability"
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
    colors = ['#004D7A', '#D55E00', '#D98CBF', '#004D7A', '#D55E00', '#D98CBF', '#004D7A', '#D55E00', '#D98CBF']  # Blue, Orange, Pink for each platform
    line_styles = ['-', '-', '-', '--', '--', '--', ':', ':', ':']  # Solid for Accel, Dashed for CPU, Dotted for DPU
    
    # Create the plot
    plt.figure(figsize=(8, 2.8))
    
    all_data = {}
    
    # Collect data for all configurations
    for i, (instance_count, machine_type, label) in enumerate(configs):
        print(f"Collecting data for {label} with {msg_size}B and {target_clients} clients...")
        latencies = collect_latency_data_for_clients(base_directory, instance_count, msg_size, machine_type, target_clients)
        
        if len(latencies) > 0:
            all_data[label] = latencies
            print(f"{label}: {len(latencies)} latency points")
            
            # Calculate and plot CDF
            x, y = calculate_cdf(latencies)
            plt.plot(x, y, color=colors[i], linewidth=3, linestyle=line_styles[i], label=label)
        else:
            print(f"No data found for {label}!")
    
    # Set labels and formatting
    plt.xlabel('Latency (μs)', fontsize=16, fontweight='medium')
    plt.ylabel('CDF', fontsize=16, fontweight='medium')
    # plt.title(f'Tail Latency CDF Comparison ({msg_size}B, {target_clients} Clients)', 
    #           fontsize=16)
    
    # Set tick font size
    plt.xticks(fontsize=16, fontweight='medium')
    plt.yticks(fontsize=16, fontweight='medium')
    
    # Add grid - only vertical lines
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Add legend in empty space
    plt.legend(prop={'size': 13, 'weight': 'medium'},bbox_to_anchor=(0.94, 0.75), loc='center right', ncol=3, framealpha=0.5, columnspacing=0.2) # Middle bottom right
    
    # Set axis limits
    plt.xlim(0, 650)
    plt.ylim(0, 1)
    
    # Print statistics for all configurations
    for label, latencies in all_data.items():
        print(f"\n{label} Statistics:")
        print(f"  Median: {np.median(latencies):.2f} μs")
        print(f"  90th percentile: {np.percentile(latencies, 90):.2f} μs")
        print(f"  95th percentile: {np.percentile(latencies, 95):.2f} μs")
        print(f"  99th percentile: {np.percentile(latencies, 99):.2f} μs")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('tail_latency_cdf_fig_17.pdf', bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as tail_latency_cdf_fig_17.pdf")
    
    # plt.show()


if __name__ == "__main__":
    create_cdf_comparison_plot()
