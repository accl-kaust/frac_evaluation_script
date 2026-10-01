import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch
import glob
import matplotlib.ticker as ticker
import pickle

import plot_fonts  # noqa: F401  (Helvetica Neue from fonts/, medium weight everywhere)

# ---------------------------------------------------------------------------
# Data processing helpers (LogProcessor, RawLatencyProcessor, the cache
# reader/writer and process_all_configurations_cached) used to live in
# topk_instances_comparison_with_DPU.py; they are inlined here verbatim so
# this script is self-contained.
# ---------------------------------------------------------------------------

class LogProcessor:
    def __init__(self, file):
        self.file = file
        self.data = []
        self.avg_latency = 0
        self.median_latency = 0
        self.percentile_95 = 0
        self.percentile_99 = 0
        
        # Extract details from filename
        match_thread = re.search(r'n_(\d+)', file)
        match_msg_size = re.search(r'm_(\d+)', file)
        match_function = re.search(r'f_(\d+)', file)
        
        self.thread_count = int(match_thread.group(1)) if match_thread else 0
        self.msg_size = int(match_msg_size.group(1)) if match_msg_size else 0
        self.function = int(match_function.group(1)) if match_function else 0

        # Regular expressions to parse log lines
        self.data_pattern = re.compile(
            r"(\d+)\s+RR\s+\.(\d+)\s+min=(\d+\.\d+)us\s+avg=(\d+\.\d+)us\s+max=(\d+\.\d+)us\s+"
            r"read\(Gbits/sec\)=(\d+\.\d+)\s+write\(Gbits/sec\)=(\d+\.\d+)\s+count=(\d+)"
        )

    def parse_log(self):
        # Parse the log file for latency data
        with open(self.file, "r") as fd:
            for line in fd:
                data_match = self.data_pattern.match(line.strip())
                if data_match:
                    second, thread, min_latency, avg_latency, max_latency, read_tp, write_tp, count = data_match.groups()
                    time = int(second)
                    if time >= 10:  # Filter data after 10 seconds (steady state)
                        self.data.append({"second": time, "avg": float(avg_latency)})

    def calculate_average_latency(self):
        # Calculate average latency
        if not self.data:
            # print(f"No data after 10s for {self.file}")
            return False
        
        latencies = [entry["avg"] for entry in self.data]
        self.avg_latency = sum(latencies) / len(latencies)
        
        # Calculate median and percentiles correctly using numpy
        self.median_latency = np.median(latencies)
        self.percentile_95 = np.percentile(latencies, 95)
        self.percentile_99 = np.percentile(latencies, 99)
        
        return True


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
            # print(f"No .txt files found in directory: {self.directory_path}")
            return False
        
        # print(f"Found {len(txt_files)} .txt files in {self.directory_path}")
        
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
                skip_count = total_lines // 4
                
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
                
                #print(f"  File {os.path.basename(txt_file)}: {total_lines} total lines, skipped first {skip_count}, used {len(file_latencies_ns)}")
                all_latencies_ns.extend(file_latencies_ns)
            
            if len(all_latencies_ns) == 0:
                print(f"No valid latency data found in directory: {self.directory_path}")
                return False
            
            # Convert from nanoseconds to microseconds and store for external access
            self.raw_latencies_us = [lat / 1000.0 for lat in all_latencies_ns]
            
            # Calculate median and 99th percentile
            self.median_latency = np.median(self.raw_latencies_us)
            self.percentile_99 = np.percentile(self.raw_latencies_us, 99)
            
            # print(f"  Combined total: {len(self.raw_latencies_us)} latency data points")
            # print(f"  Thread count: {self.thread_count}, Msg size: {self.msg_size}")
            # print(f"  Median: {self.median_latency:.2f}μs, 99th percentile: {self.percentile_99:.2f}μs")
            
            return True
            
        except Exception as e:
            # print(f"Error processing files in {self.directory_path}: {e}")
            return False


def get_log_files_for_config(directory, function_patterns, msg_size=1024, machine_type=1):
    """Get log files for specific functions and message size"""
    files = []
    
    # For topk_only_1_inst, files have different patterns
    if "top_k_1_inst" in directory:
        if machine_type == 1:  # FRAC (O_1)
            for pattern in function_patterns:
                file_pattern = f"rr_d_*_m_{msg_size}_n_*_C_1_f_{pattern}_O_1.log"
                matching_files = glob.glob(os.path.join(directory, file_pattern))
                files.extend(matching_files)
        elif machine_type == 2:  # CPU (O_2)
            # Try multiple pattern variations to find CPU files
            pattern_variations = [
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_X_{msg_size}_O_2.log",  # Format in topk_only_1_inst
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_*_O_2.log",             # Any pattern in between
                f"rr_d_*_m_{msg_size}_n_*_*_f_1_*_O_2.log",               # More flexible pattern
                f"*_m_{msg_size}_*_f_1_*_O_2.log"                         # Most flexible pattern
            ]
            
            for file_pattern in pattern_variations:
                matching_files = glob.glob(os.path.join(directory, file_pattern))
                files.extend(matching_files)
        elif machine_type == 3:  # DPU (O_3) - follows CPU patterns
            pattern_variations = [
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_X_{msg_size}_O_3.log",
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_*_O_3.log",
                f"rr_d_*_m_{msg_size}_n_*_*_f_1_*_O_3.log",
                f"*_m_{msg_size}_*_f_1_*_O_3.log"
            ]
            for file_pattern in pattern_variations:
                matching_files = glob.glob(os.path.join(directory, file_pattern))
                files.extend(matching_files)
    else:
        # For 2_inst and 4_inst
        if machine_type == 1:  # FRAC (O_1)
            file_pattern = f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_R_64_X_{msg_size}_O_1.log"
            matching_files = glob.glob(os.path.join(directory, file_pattern))
            files.extend(matching_files)
        elif machine_type == 2:  # CPU (O_2)
            # Try multiple pattern variations for CPU files
            pattern_variations = [
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_X_{msg_size}_O_2.log",  # Format with X in between
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_*_O_2.log",             # Any pattern in between
                f"rr_d_*_m_{msg_size}_n_*_*_f_1_*_O_2.log",               # More flexible pattern
                f"*_m_{msg_size}_*_f_1_*_O_2.log"                         # Most flexible pattern
            ]
            
            for file_pattern in pattern_variations:
                matching_files = glob.glob(os.path.join(directory, file_pattern))
                files.extend(matching_files)
        elif machine_type == 3:  # DPU (O_3) - follows CPU patterns
            pattern_variations = [
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_X_{msg_size}_O_3.log",
                f"rr_d_*_m_{msg_size}_n_*_C_1_f_1_*_O_3.log",
                f"rr_d_*_m_{msg_size}_n_*_*_f_1_*_O_3.log",
                f"*_m_{msg_size}_*_f_1_*_O_3.log"
            ]
            for file_pattern in pattern_variations:
                matching_files = glob.glob(os.path.join(directory, file_pattern))
                files.extend(matching_files)
    
    # Remove duplicates (in case multiple patterns matched the same file)
    files = list(set(files))
    
    if len(files) == 0:
        # print(f"WARNING: No matching files found!")
        pass
    
    return sorted(files)


def get_tail_latency_directories(base_directory, instance_count, msg_size=1024, machine_type=1):
    """Get directories containing raw tail latency data for O_1 (FRAC) or O_2 (CPU)"""
    directory = f"{base_directory}/top_k_{instance_count}_inst"
    
    if not os.path.exists(directory):
        # print(f"Directory not found: {directory}")
        return []
    
    # Look for directories matching the pattern based on machine type
    if machine_type == 1:  # FRAC (O_1)
        dir_pattern = f"rr_d_*_m_{msg_size}_n_*_f_1_O_1"
    elif machine_type == 2:  # CPU (O_2)
        dir_pattern = f"rr_d_*_m_{msg_size}_n_*_f_1_O_2"
    elif machine_type == 3:  # DPU (O_3)
        dir_pattern = f"rr_d_*_m_{msg_size}_n_*_f_1_O_3"
    
    matching_dirs = glob.glob(os.path.join(directory, dir_pattern))
    
    return sorted(matching_dirs)


def process_data_for_instance_count(base_directory, instance_count, functions, client_multiplier, msg_size=1024, machine_type=1, use_tail_data=False):
    """Process logs for specific instance configuration"""
    
    if use_tail_data:  # Use new tail latency data for both FRAC (O_1) and CPU (O_2)
        directories = get_tail_latency_directories(base_directory, instance_count, msg_size, machine_type)
        
        machine_str = (
            "FRAC (O_1) - Tail Data" if machine_type == 1 else
            ("CPU (O_2) - Tail Data" if machine_type == 2 else "DPU (O_3) - Tail Data")
        )
        #print(f"\nProcessing tail latency data for {instance_count} instances with message size {msg_size}B - {machine_str}")
        #print(f"Found {len(directories)} data directories")
        
        # Group directories by thread count
        thread_groups = {}
        for dir_path in directories:
            processor = RawLatencyProcessor(dir_path)
            if processor.thread_count not in thread_groups:
                thread_groups[processor.thread_count] = []
            thread_groups[processor.thread_count].append(dir_path)
        
        results = []
        for thread_count, dir_group in sorted(thread_groups.items()):
            #print(f"Processing thread count {thread_count} with {len(dir_group)} directories")
            
            # Calculate client number - for CPU (O_2) don't multiply by instance count
            if machine_type == 1:  # FRAC
                client_number = thread_count * client_multiplier
            else:  # CPU (O_2)
                client_number = thread_count  # No multiplication for CPU
            
            # Collect all latency data from all directories for this thread count
            all_median_latencies = []
            all_p99_latencies = []
            
            for dir_path in dir_group:
                processor = RawLatencyProcessor(dir_path)
                if processor.process_raw_latency_data():
                    all_median_latencies.append(processor.median_latency)
                    all_p99_latencies.append(processor.percentile_99)
            
            if all_median_latencies and all_p99_latencies:
                # Collect all raw latency data from all directories for this thread count
                all_raw_latencies = []
                for dir_path in dir_group:
                    processor = RawLatencyProcessor(dir_path)
                    if processor.process_raw_latency_data():
                        # Use the already processed raw latency data (no duplicate file processing)
                        all_raw_latencies.extend(processor.raw_latencies_us)
                
                if all_raw_latencies:
                    # Calculate all required percentiles from combined raw data
                    combined_p25 = np.percentile(all_raw_latencies, 25)
                    combined_median = np.percentile(all_raw_latencies, 50)
                    combined_p75 = np.percentile(all_raw_latencies, 75)
                    combined_p90 = np.percentile(all_raw_latencies, 90)
                    combined_p95 = np.percentile(all_raw_latencies, 95)
                    combined_p99 = np.percentile(all_raw_latencies, 99)
                    
                    #print(f"  Combined: median latency = {combined_median:.2f}μs, 99th percentile = {combined_p99:.2f}μs")
                    #print(f"  Client number: {client_number}")
                    
                    # Return format: (client_number, p25, median, p75, p90, p95, p99)
                    results.append((client_number, combined_p25, combined_median, combined_p75, combined_p90, combined_p95, combined_p99))
                else:
                    # Fallback to using pre-calculated values if raw data processing fails
                    combined_median = np.median(all_median_latencies)
                    combined_p99 = np.median(all_p99_latencies)  # Use median of 99th percentiles
                    results.append((client_number, combined_median, combined_median, combined_median, combined_median, combined_median, combined_p99))
            else:
                print(f"  No valid data for thread count {thread_count}")
        
        return sorted(results)
    
    else:  # Use original data processing for CPU (O_2) or when not using tail data
        directory = f"{base_directory}/topk_only_{instance_count}_inst"
        files = get_log_files_for_config(directory, functions, msg_size, machine_type)
        
        machine_str = "FRAC (O_1)" if machine_type == 1 else "CPU (O_2)"
        
        # Group files by thread count
        thread_groups = {}
        for file in files:
            processor = LogProcessor(file)
            if processor.thread_count not in thread_groups:
                thread_groups[processor.thread_count] = []
            thread_groups[processor.thread_count].append(file)
        
        results = []
        for thread_count, file_group in sorted(thread_groups.items()):
            
            # Calculate client number - for CPU (O_2) don't multiply by instance count
            if machine_type == 1:  # FRAC
                client_number = thread_count * client_multiplier
            else:  # CPU (O_2)
                client_number = thread_count  # No multiplication for CPU
            
            # Combine all latency data points from all files for this thread count
            all_latency_data = []
            for file in file_group:
                processor = LogProcessor(file)
                processor.parse_log()
                if processor.data:
                    # Extract all individual latency values and add them to the combined list
                    file_latencies = [entry["avg"] for entry in processor.data]
                    all_latency_data.extend(file_latencies)
            
            if all_latency_data:
                # Calculate median and percentiles from the combined data
                combined_p25 = np.percentile(all_latency_data, 25)
                combined_median = np.percentile(all_latency_data, 50)
                combined_p75 = np.percentile(all_latency_data, 75)
                combined_p90 = np.percentile(all_latency_data, 90)
                combined_p95 = np.percentile(all_latency_data, 95)
                combined_p99 = np.percentile(all_latency_data, 99)
                
                results.append((client_number, combined_p25, combined_median, combined_p75, combined_p90, combined_p95, combined_p99))
        
        return sorted(results)


def save_processed_data(data_dict, cache_file_path):
    """Save processed data to cache file"""
    try:
        with open(cache_file_path, 'w') as f:
            f.write("# Cached latency data\n")
            f.write("# Format: config_name,msg_size,clients,p25,median,p75,p90,p95,p99\n")
            for config_name, msg_data in data_dict.items():
                for msg_size, client_data in msg_data.items():
                    for clients, p25, median, p75, p90, p95, p99 in client_data:
                        f.write(f"{config_name},{msg_size},{clients},{p25:.3f},{median:.3f},{p75:.3f},{p90:.3f},{p95:.3f},{p99:.3f}\n")
        print(f"Cached processed data to {cache_file_path}")
        return True
    except Exception as e:
        print(f"Error saving cache file: {e}")
        return False


def load_processed_data(cache_file_path):
    """Load processed data from cache file"""
    try:
        data_dict = {}
        with open(cache_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split(',')
                if len(parts) == 5:  # Old format: config_name,msg_size,clients,median,p99
                    config_name, msg_size, clients, median, p99 = parts
                    msg_size = int(msg_size)
                    clients = int(clients)
                    median = float(median)
                    p99 = float(p99)
                    # Convert old format to new format (use median for missing percentiles)
                    data_tuple = (clients, median, median, median, median, median, p99)
                elif len(parts) == 9:  # New format: config_name,msg_size,clients,p25,median,p75,p90,p95,p99
                    config_name, msg_size, clients, p25, median, p75, p90, p95, p99 = parts
                    msg_size = int(msg_size)
                    clients = int(clients)
                    p25 = float(p25)
                    median = float(median)
                    p75 = float(p75)
                    p90 = float(p90)
                    p95 = float(p95)
                    p99 = float(p99)
                    data_tuple = (clients, p25, median, p75, p90, p95, p99)
                else:
                    continue
                
                if config_name not in data_dict:
                    data_dict[config_name] = {}
                if msg_size not in data_dict[config_name]:
                    data_dict[config_name][msg_size] = []
                
                data_dict[config_name][msg_size].append(data_tuple)
        
        print(f"Loaded cached data from {cache_file_path}")
        return data_dict
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return None


def print_cached_data(data_dict):
    """Print all cached percentile data in a readable format"""
    print("\n" + "="*80)
    print("LOADED CACHED PERCENTILE DATA")
    print("="*80)
    
    msg_sizes = [1024, 4096]
    
    for msg_size in msg_sizes:
        print(f"\n{'='*20} {msg_size}B Request Size {'='*20}")
        
        # Print table header
        print(f"\n{'Configuration':<15} {'Clients':<8} {'P25':<8} {'Median':<8} {'P75':<8} {'P90':<8} {'P95':<8} {'P99':<8}")
        print("-" * 75)
        
        # Print data for each configuration
        config_display_names = {
            "1_accel": "1 Accel",
            "2_accel": "2 Accel", 
            "4_accel": "4 Accel",
            "1_cpu": "1 CPU",
            "2_cpu": "2 CPU",
            "4_cpu": "4 CPU",
            "1_dpu": "1 DPU",
            "2_dpu": "2 DPU",
            "4_dpu": "4 DPU"
        }
        
        for config_name in ["1_accel", "2_accel", "4_accel", "1_cpu", "2_cpu", "4_cpu", "1_dpu", "2_dpu", "4_dpu"]:
            display_name = config_display_names[config_name]
            if config_name in data_dict and msg_size in data_dict[config_name]:
                data = data_dict[config_name][msg_size]
                if data:
                    for entry in data:
                        if len(entry) == 7:  # New format: (clients, p25, median, p75, p90, p95, p99)
                            clients, p25, median, p75, p90, p95, p99 = entry
                            print(f"{display_name:<15} {clients:<8} {p25:<8.2f} {median:<8.2f} {p75:<8.2f} {p90:<8.2f} {p95:<8.2f} {p99:<8.2f}")
                        else:  # Old format: (clients, median, p99) - convert for display
                            clients, median, p99 = entry
                            print(f"{display_name:<15} {clients:<8} {median:<8.2f} {median:<8.2f} {median:<8.2f} {median:<8.2f} {median:<8.2f} {p99:<8.2f}")
                else:
                    print(f"{display_name:<15} {'No data'}")
            else:
                print(f"{display_name:<15} {'No data'}")


def process_all_configurations_cached(base_directory):
    """Process all configurations with caching support"""
    cache_file = os.path.join(base_directory, "processed_latency_cache.txt")
    
    # Define the required configuration matrix (including DPU O_3)
    configurations = [
        ("1_accel", 1, 1, 1),    # name, instance_count, client_multiplier, machine_type
        ("2_accel", 2, 2, 1),
        ("4_accel", 4, 4, 1),
        ("1_cpu", 1, 1, 2),
        ("2_cpu", 2, 2, 2),
        ("4_cpu", 4, 4, 2),
        ("1_dpu", 1, 1, 3),
        ("2_dpu", 2, 2, 3),
        ("4_dpu", 4, 4, 3)
    ]
    msg_sizes = [1024, 4096]
    
    # Try to load from cache first; if incomplete, fill only missing entries
    if os.path.exists(cache_file):
        cached_data = load_processed_data(cache_file)
        if cached_data is None:
            cached_data = {}
        
        missing = []
        for config_name, instance_count, client_multiplier, machine_type in configurations:
            for msg_size in msg_sizes:
                if config_name not in cached_data or msg_size not in cached_data[config_name] or not cached_data[config_name][msg_size]:
                    missing.append((config_name, instance_count, client_multiplier, machine_type, msg_size))
        
        if not missing and cached_data:
            print_cached_data(cached_data)
            return cached_data
        
        print("Cache incomplete or invalid, processing missing entries...")
        for config_name, instance_count, client_multiplier, machine_type, msg_size in missing:
            print(f"Processing {config_name} for {msg_size}B...")
            data = process_data_for_instance_count(base_directory, instance_count, ['1'], 
                                                 client_multiplier, msg_size, machine_type, use_tail_data=True)
            if config_name not in cached_data:
                cached_data[config_name] = {}
            cached_data[config_name][msg_size] = data
        
        save_processed_data(cached_data, cache_file)
        print_cached_data(cached_data)
        return cached_data
    
    # No cache; process all required entries
    print("Cache not found, processing data for all required entries...")
    data_dict = {}
    for config_name, instance_count, client_multiplier, machine_type in configurations:
        data_dict[config_name] = {}
        for msg_size in msg_sizes:
            print(f"Processing {config_name} for {msg_size}B...")
            data = process_data_for_instance_count(base_directory, instance_count, ['1'], 
                                                 client_multiplier, msg_size, machine_type, use_tail_data=True)
            data_dict[config_name][msg_size] = data
    save_processed_data(data_dict, cache_file)
    print_cached_data(data_dict)
    return data_dict





def create_separate_accel_plots(base_directory):
    """Create 6 plots in one row - pairs of (1024B, 4096B) for each accelerator count"""
    
    # Use cached data processing
    data_dict = process_all_configurations_cached(base_directory)
    
    # Create figure with 6 subplots horizontally
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5), sharey=True)
    
    # Colors - one color per device type
    color_accel = '#004D7A'  # Blue for Accel
    color_cpu = '#D55E00'    # Orange for CPU
    color_dpu = '#D98CBF'    # Pink for DPU
    
    # Configuration: (accel_count, accel_key, cpu_key, dpu_key, msg_size, title)
    plot_configs = [
    (1, "1_accel", "1_cpu", "1_dpu", 1024, "1 Acc/Core - 1024B"),
    (1, "1_accel", "1_cpu", "1_dpu", 4096, "1 Acc/Core - 4096B"),
    (2, "2_accel", "2_cpu", "2_dpu", 1024, "2 Accs/Cores - 1024B"),
    (2, "2_accel", "2_cpu", "2_dpu", 4096, "2 Accs/Cores - 4096B"),
    (4, "4_accel", "4_cpu", "4_dpu", 1024, "4 Accs/Cores - 1024B"),
    (4, "4_accel", "4_cpu", "4_dpu", 4096, "4 Accs/Cores - 4096B"),
    ]
    
    # Collect all latencies to set consistent y-axis limits
    all_latencies = []
    
    for idx, (accel_count, accel_key, cpu_key, dpu_key, msg_size, title) in enumerate(plot_configs):
        ax = axes[idx]
        
        # Get data for this configuration
        data_accel = data_dict.get(accel_key, {}).get(msg_size, [])
        data_cpu = data_dict.get(cpu_key, {}).get(msg_size, [])
        data_dpu = data_dict.get(dpu_key, {}).get(msg_size, [])
        
        # For idx 1 (1 Accel - 4096B), filter out client 2 completely
        if idx == 1:
            data_accel = [x for x in data_accel if x[0] != 2]
            data_cpu = [x for x in data_cpu if x[0] != 2]
            data_dpu = [x for x in data_dpu if x[0] != 2]
        
        # For idx 2, 3 (2 Accels/Cores), filter out clients 1 and 2 completely
        if idx in [2, 3]:
            data_accel = [x for x in data_accel if x[0] not in [1, 2]]
            data_cpu = [x for x in data_cpu if x[0] not in [1, 2]]
            data_dpu = [x for x in data_dpu if x[0] not in [1, 2]]
        
        # For idx 4, 5 (4 Accels/Cores), filter out client 1 completely
        if idx in [4, 5]:
            data_accel = [x for x in data_accel if x[0] != 1]
            data_cpu = [x for x in data_cpu if x[0] != 1]
            data_dpu = [x for x in data_dpu if x[0] != 1]
        
        # Extract data points
        clients_accel = [x[0] for x in data_accel]
        p25_accel = [x[1] if len(x) == 7 else x[1] for x in data_accel]
        latency_accel = [x[2] if len(x) == 7 else x[1] for x in data_accel]
        p75_accel = [x[3] if len(x) == 7 else x[1] for x in data_accel]
        
        clients_cpu = [x[0] for x in data_cpu]
        p25_cpu = [x[1] if len(x) == 7 else x[1] for x in data_cpu]
        latency_cpu = [x[2] if len(x) == 7 else x[1] for x in data_cpu]
        p75_cpu = [x[3] if len(x) == 7 else x[1] for x in data_cpu]
        
        clients_dpu = [x[0] for x in data_dpu]
        p25_dpu = [x[1] if len(x) == 7 else x[1] for x in data_dpu]
        latency_dpu = [x[2] if len(x) == 7 else x[1] for x in data_dpu]
        p75_dpu = [x[3] if len(x) == 7 else x[1] for x in data_dpu]
        
        # Collect all latencies for y-axis limits
        all_latencies.extend(latency_accel + latency_cpu + latency_dpu)
        all_latencies.extend(p25_accel + p75_accel + p25_cpu + p75_cpu + p25_dpu + p75_dpu)
        
        # Plot Accel (blue)
        if clients_accel and latency_accel:
            ax.fill_between(clients_accel, p25_accel, p75_accel, 
                           alpha=0.3, color=color_accel)
            ax.plot(clients_accel, latency_accel, marker='s', linestyle='-', 
                   color=color_accel, linewidth=2, markersize=8)
        
        # Plot CPU (orange)
        if clients_cpu and latency_cpu:
            ax.fill_between(clients_cpu, p25_cpu, p75_cpu, 
                           alpha=0.2, color=color_cpu)
            ax.plot(clients_cpu, latency_cpu, marker='^', linestyle='-', 
                   color=color_cpu, linewidth=2, markersize=8)
        
        # Plot DPU (pink)
        if clients_dpu and latency_dpu:
            ax.fill_between(clients_dpu, p25_dpu, p75_dpu, 
                           alpha=0.2, color=color_dpu)
            ax.plot(clients_dpu, latency_dpu, marker='o', linestyle='-', 
                   color=color_dpu, linewidth=2, markersize=6)
        
        # Set title
        ax.set_title(title, fontsize=16, fontweight='medium')
        
        # Set x-axis label
        ax.set_xlabel('Clients', fontsize=16, fontweight='medium')
        
        # Turn off all grid lines
        ax.grid(False)
        ax.xaxis.grid(False, which='both')
        ax.yaxis.grid(False, which='both')
        
        # Set x-ticks
        all_clients = set()
        for data in [data_accel, data_cpu, data_dpu]:
            for x in data:
                all_clients.add(x[0])
        ax.set_xticks(sorted(list(all_clients)))
        
        # Set tick font size
        ax.tick_params(axis='both', which='major', labelsize=14)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('medium')
        
        # Add annotations for ALL points
        def annotate_all_points(clients, latencies, color, y_multiplier, va='top'):
            for i, client_count in enumerate(clients):
                y_pos = latencies[i] * y_multiplier
                if client_count == max(clients):  # Last point
                    x_pos = client_count
                    ha = 'right'
                elif client_count == min(clients):  # First point
                    x_pos = client_count + 0.3
                    ha = 'left'
                else:  # Middle points
                    x_pos = client_count
                    ha = 'center'
                ax.annotate(f'{int(latencies[i])}', 
                           xy=(client_count, latencies[i]), 
                           xytext=(x_pos, y_pos),
                           fontsize=14, fontweight='medium', color=color,
                           ha=ha, va=va)
        
        # Special handling for first two figures (1 Accel) - CPU and DPU lines are very close
        if idx == 0:  # 1 Accel - 1024B
            if clients_accel and latency_accel:
                annotate_all_points(clients_accel, latency_accel, color_accel, y_multiplier=0.68, va='top')
            if clients_cpu and latency_cpu:
                annotate_all_points(clients_cpu, latency_cpu, color_cpu, y_multiplier=1.15, va='bottom')
            if clients_dpu and latency_dpu:
                # Skip DPU annotations for clients 1, 2, 4 to avoid overlap
                clients_dpu_filtered = [c for c in clients_dpu if c not in [1, 2, 4]]
                latency_dpu_filtered = [latency_dpu[i] for i, c in enumerate(clients_dpu) if c not in [1, 2, 4]]
                annotate_all_points(clients_dpu_filtered, latency_dpu_filtered, color_dpu, y_multiplier=1.8, va='bottom')
        elif idx == 1:  # 1 Accel - 4096B - client 2 already filtered from data
            if clients_accel and latency_accel:
                annotate_all_points(clients_accel, latency_accel, color_accel, y_multiplier=0.68, va='top')
            if clients_cpu and latency_cpu:
                annotate_all_points(clients_cpu, latency_cpu, color_cpu, y_multiplier=1.15, va='bottom')
            if clients_dpu and latency_dpu:
                # Skip DPU annotations for clients 1, 4 to avoid overlap (2 already filtered)
                clients_dpu_filtered = [c for c in clients_dpu if c not in [1, 4]]
                latency_dpu_filtered = [latency_dpu[i] for i, c in enumerate(clients_dpu) if c not in [1, 4]]
                annotate_all_points(clients_dpu_filtered, latency_dpu_filtered, color_dpu, y_multiplier=2, va='bottom')
        else:
            # Accel below, CPU above, DPU higher above (staggered to avoid overlap)
            if clients_accel and latency_accel:
                annotate_all_points(clients_accel, latency_accel, color_accel, y_multiplier=0.72, va='top')
            if clients_cpu and latency_cpu:
                annotate_all_points(clients_cpu, latency_cpu, color_cpu, y_multiplier=1.20, va='bottom')
            if clients_dpu and latency_dpu:
                annotate_all_points(clients_dpu, latency_dpu, color_dpu, y_multiplier=1.45, va='bottom')
    
    # Set log scale and consistent y-axis limits AFTER all data is plotted
    if all_latencies:
        y_min = min(all_latencies) * 0.6
        y_max = max(all_latencies) * 2.5
        
        for ax in axes:
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)
            # Disable y-axis grid after log scale is set
            ax.yaxis.grid(False, which='both')
            ax.tick_params(axis='y', which='minor', left=False, right=False)
    
    # Set y-axis label only for left subplot
    axes[0].set_ylabel('Latency (μs)', fontsize=16, fontweight='medium')
    
    # Remove y-axis ticks from all but first subplot
    for ax in axes[1:]:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Create shared legend at the top
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color=color_accel, linestyle='-', linewidth=2, markersize=10, label='fRAC'),
        plt.Line2D([0], [0], marker='^', color=color_cpu, linestyle='-', linewidth=2, markersize=10, label='CPU'),
        plt.Line2D([0], [0], marker='o', color=color_dpu, linestyle='-', linewidth=2, markersize=8, label='DPU'),
    ]
    fig.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, 1.02),
        loc='upper center',
        ncol=3,
        frameon=False,
        prop={'size': 16, 'weight': 'medium'},
        columnspacing=2.0,
        handletextpad=0.5
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save figure
    plt.savefig('scalability_fig_16.pdf', bbox_inches='tight', dpi=300)
    print("\nPlot saved as scalability_fig_16.pdf")


if __name__ == "__main__":
    base_directory = "data/scalability"
    create_separate_accel_plots(base_directory)
