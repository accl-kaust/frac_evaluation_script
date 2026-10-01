"""Latency vs application throughput, with and without fRAC (new data, 30-09-2026).

Data: data/latency_throughput/rr_d_30_m_4096_n_<clients>_C_1_f_0_O_<0|1>.log
  O_0 = without fRAC (plain TCP stack), O_1 = with fRAC.
  Each log is one 30 s run (seconds 0-29) with one "RR .<thread>" line per
  client thread per second and one "RR Total-Throughput" line per second.

Per log file (one client count n):
  latency    = median of the per-thread "avg=" values from the last 20 s
               (seconds 10-29); shaded band = 25th-75th percentile of the same values
  throughput = mean of the per-second "Total-Throughput write(Gbits/sec)" over the same 20 s

Output: latency_throughput_fig_13.pdf
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import plot_fonts  # noqa: F401  (Helvetica Neue from fonts/, medium weight everywhere)

DATA_DIR = "data/latency_throughput"
FILE_PREFIX = "rr_d_30_m_4096_n_"   # 30 s runs, 4096 B requests
RUN_SECONDS = 30
KEEP_LAST_SECONDS = 20               # analyse seconds 10-29 only
FIRST_SECOND = RUN_SECONDS - KEEP_LAST_SECONDS

# Log line formats
DATA_PATTERN = re.compile(
    r"(\d+)\s+RR\s+\.(\d+)\s+min=(\d+\.\d+)us\s+avg=(\d+\.\d+)us\s+max=(\d+\.\d+)us\s+"
    r"read\(Gbits/sec\)=(\d+\.\d+)\s+write\(Gbits/sec\)=(\d+\.\d+)\s+count=(\d+)"
)
TP_PATTERN = re.compile(
    r"(\d+)\s+RR\s+Total-Throughput\s+read\(Gbits/sec\)=(\d+\.\d+)\s+write\(Gbits/sec\)=(\d+\.\d+)"
)


class LogProcessor:
    """Parses one fperf log and summarises the last KEEP_LAST_SECONDS seconds."""

    def __init__(self, file):
        self.file = file
        match = re.search(r'n_(\d+)', os.path.basename(file))
        self.thread_count = int(match.group(1)) if match else 0
        self.avg_samples = []        # per-thread avg latency (us), one per thread per second
        self.write_tp_samples = []   # total write throughput (Gbit/s), one per second
        self.median_latency = 0.0
        self.p25_latency = 0.0
        self.p75_latency = 0.0
        self.throughput = 0.0

    def parse_log(self):
        with open(self.file, "r") as fd:
            for line in fd:
                line = line.strip()
                m = DATA_PATTERN.match(line)
                if m:
                    second, avg_latency = int(m.group(1)), float(m.group(4))
                    if second >= FIRST_SECOND:
                        self.avg_samples.append(avg_latency)
                    continue
                m = TP_PATTERN.match(line)
                if m:
                    second, write_tp = int(m.group(1)), float(m.group(3))
                    if second >= FIRST_SECOND:
                        self.write_tp_samples.append(write_tp)

    def summarize(self):
        if not self.avg_samples or not self.write_tp_samples:
            print(f"No data in seconds {FIRST_SECOND}-{RUN_SECONDS - 1} for {self.file}")
            return False
        samples = np.array(self.avg_samples)
        self.median_latency = float(np.median(samples))
        self.p25_latency = float(np.percentile(samples, 25))
        self.p75_latency = float(np.percentile(samples, 75))
        self.throughput = float(np.mean(self.write_tp_samples))
        return True


def get_log_files(directory, suffix):
    """Log files in `directory` (not subdirectories) for one option, e.g. suffix 'O_1.log'."""
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.startswith(FILE_PREFIX) and f.endswith(suffix)
    )


def collect(directory, suffix):
    """List of (throughput, median, p25, p75, clients) for every log with the given suffix."""
    points = []
    for log_file in get_log_files(directory, suffix):
        proc = LogProcessor(log_file)
        proc.parse_log()
        if proc.summarize():
            points.append((proc.throughput, proc.median_latency, proc.p25_latency,
                           proc.p75_latency, proc.thread_count, len(proc.avg_samples)))
    return sorted(points)  # by throughput, for line plotting


def print_table(name, points):
    print(f"\n{name}:")
    print(f"  {'n':>4s} {'Gbps':>8s} {'p25':>8s} {'median':>8s} {'p75':>8s} {'samples':>8s}")
    for tp, med, p25, p75, n, cnt in sorted(points, key=lambda p: p[4]):
        print(f"  {n:4d} {tp:8.2f} {p25:8.2f} {med:8.2f} {p75:8.2f} {cnt:8d}")


def plot_throughput_vs_latency(with_frac, without_frac):
    plt.figure(figsize=(8, 4.5))

    for points, marker, color, label, alpha in (
        (with_frac, 'o-', 'blue', 'With fRAC', 0.2),
        (without_frac, '^-', '#D55E00', 'Without fRAC', 0.35),
    ):
        x = [p[0] for p in points]
        med = [p[1] for p in points]
        p25 = [p[2] for p in points]
        p75 = [p[3] for p in points]
        plt.plot(x, med, marker, color=color, label=label, markersize=10, linewidth=2)
        if len(x) > 2:
            plt.fill_between(x, p25, p75, color=color, alpha=alpha)  # 25th-75th percentile

    plt.xlabel('Application Throughput (Gbps)', fontsize=20, fontweight='medium')
    plt.ylabel('Latency (μs)', fontsize=20, fontweight='medium')
    plt.xticks(fontsize=20, fontweight='medium')
    plt.yticks(fontsize=20, fontweight='medium')
    plt.legend(prop={'size': 20, 'weight': 'medium'}, loc='upper left', ncol=1,
               bbox_to_anchor=(0, 1.02), frameon=False, columnspacing=0.5, markerscale=1.3)
    plt.tight_layout()
    plt.savefig('latency_throughput_fig_13.pdf', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    with_frac = collect(DATA_DIR, "O_1.log")        # O_1 = with fRAC
    without_frac = collect(DATA_DIR, "O_0.log")     # O_0 = without fRAC

    print(f"Seconds {FIRST_SECOND}-{RUN_SECONDS - 1} of each run; latency = median of per-thread avg, "
          f"band = p25-p75, throughput = mean of per-second total write")
    print_table("With fRAC (O_1)", with_frac)
    print_table("Without fRAC (O_0)", without_frac)

    if with_frac and without_frac:
        plot_throughput_vs_latency(with_frac, without_frac)
        print("\nPlot saved as latency_throughput_fig_13.pdf")
    else:
        print("Not enough data to create a plot")
