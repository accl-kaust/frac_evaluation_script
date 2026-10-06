#!/usr/bin/env python3
"""
Reproduce the "ClickNP_Result_no_shadow" latency figure.

This is a standalone rewrite of the plotting cells in
``Networked experiment.ipynb`` (cells 8, 9, 10, 16 and 17).
It compares the end-to-end latency of two offloading architectures
across seven payload sizes (32B ... 2048B):

    * "Traditional offload"  <-  data/Networked_CPU_control_no_buffer/
                                 (notebook variable: Networked_multi_CPU, plot_lists[5])
    * "Direct offload"       <-  data/Networked_multi/
                                 (notebook variable: Networked_multi, plot_lists[4])

Each data file ``<flits>.txt`` holds one measurement per line:
``<sequence id> <latency in FPGA clock cycles>``.  Each flit carries 8 B, so
4, 8, 16, 32, 64, 128, 256 flits correspond to 32B ... 2048B.

Two layouts of the same data are available (``--variant``):

    wide    (default)  10 x 2.5 in, categorical x axis (Figure 3).  Same
                       layout as the notebook cell that wrote
                       ClickNP_Result_no_shadow_log_scale.pdf (the y axis is
                       actually linear despite that file name).
    square             7 x 4 in, log2 x axis.  Same layout as the
                       notebook cell that wrote ClickNP_Result_no_shadow_square.pdf.

Fonts, colours and sizes follow the shared style in ``plot_fonts``: Helvetica Neue
Medium, the paper palette (baseline = vermilion triangles, direct offload = dark-blue
squares) and line/marker/text sizes chosen so that, printed at 0.95\columnwidth
(Figure 3), they come out identical to the other figures.  The notebook's 3 pt lines
and black-outlined scatter overlay are intentionally not reproduced.

Usage:
    python clicknp_fig_3.py                 # wide -> ClickNP_Result_no_shadow.pdf
    python clicknp_fig_3.py --variant square
    python clicknp_fig_3.py --variant both --png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import plot_fonts  # noqa: E402  (Helvetica Neue from fonts/, shared palette and print-size style)


# --------------------------------------------------------------------------- #
# Experiment constants (from the notebook)
# --------------------------------------------------------------------------- #
CLOCK_MHZ = 175                      # FPGA clock, cycles / us
FLITS = [4, 8, 16, 32, 64, 128, 256]  # payload size in flits (8 B each)
X_LABELS = ["32B", "64B", "128B", "256B", "512B", "1024B", "2048B"]
MAX_SAMPLES = 1000                   # the notebook keeps the first 1000 unique samples

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# series name -> (sub-folder in data dir, line colour, marker)
# Colour and marker by role, shared with figs 13 / 16: baseline = vermilion triangle,
# ours (direct offload) = dark-blue square.
SERIES = {
    "Traditional offload": ("Networked_CPU_control_no_buffer", plot_fonts.C_BASE, plot_fonts.M_BASE),
    "Direct offload": ("Networked_multi", plot_fonts.C_OURS, plot_fonts.M_OURS),
}

# Figure 3 is printed at 0.95\columnwidth = 228 pt; the tight-cropped wide PDF is ~625 pt.
WIDE_PRINTED_WIDTH_PT = 228.0
WIDE_CROPPED_WIDTH_PT = 625.1


# --------------------------------------------------------------------------- #
# Data loading (faithful to notebook cells 8 / 9 / 10)
# --------------------------------------------------------------------------- #
def load_latency_cycles(path: str) -> list[int]:
    """Read one ``<flits>.txt`` file and return the latency samples in cycles.

    Mirrors the notebook: rows are de-duplicated on consecutive identical
    latency values, zero latencies are dropped, and only the first
    ``MAX_SAMPLES`` rows are kept.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            rows.append([int(v) for v in parts])

    unique_rows = []
    prev = None
    for row in rows:
        if len(row) < 2:
            continue
        if row[1] != prev and row[1] != 0:
            unique_rows.append(row)
            prev = row[1]
    unique_rows = unique_rows[:MAX_SAMPLES]
    return [row[1] for row in unique_rows]


def summarize_us(samples_cycles: list[int]) -> dict[str, float]:
    """Median / percentiles in microseconds, rounded the same way as the notebook
    (round to whole cycles first, then to 2 decimals after dividing by the clock)."""
    arr = np.asarray(samples_cycles, dtype=float)
    stats_cycles = {
        "median": round(float(np.median(arr))),
        "p90": round(float(np.percentile(arr, 90))),
        "p99.9": round(float(np.percentile(arr, 99.9))),
        "p25": round(float(np.percentile(arr, 25))),
        "p75": round(float(np.percentile(arr, 75))),
    }
    return {k: round(v / CLOCK_MHZ, 2) for k, v in stats_cycles.items()}


def load_series(data_dir: str, sub_folder: str) -> dict[str, list[float]]:
    """Return {stat_name: [value per payload size]} for one series."""
    per_size = []
    for flits in FLITS:
        path = os.path.join(data_dir, sub_folder, f"{flits}.txt")
        if not os.path.isfile(path):
            sys.exit(f"Missing data file: {path}")
        per_size.append(summarize_us(load_latency_cycles(path)))
    return {stat: [s[stat] for s in per_size] for stat in per_size[0]}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_wide(series: dict[str, dict[str, list[float]]], out_path: str,
              save_png: bool) -> None:
    """Notebook cell 17 layout: 10 x 2.5 in, categorical x axis (Figure 3 in the paper)."""
    pstyle = plot_fonts.paper_style(WIDE_PRINTED_WIDTH_PT, WIDE_CROPPED_WIDTH_PT)
    plt.figure(figsize=(10, 2.5))
    x = np.arange(len(X_LABELS))

    for label, (_, colour, marker) in SERIES.items():
        plt.plot(x, series[label]["median"], color=colour, marker=marker, label=label,
                 linewidth=pstyle.line, markersize=pstyle.marker)

    plt.yscale("linear")
    # The published PDF / notebook output has y ticks at 0, 50, ..., 200.
    # Matplotlib's automatic locator picks 0, 100, 200 for a 2.5 in tall
    # figure with 16 pt tick labels, so pin the bin count explicitly.
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(x, X_LABELS)
    plt.xlabel("Data Size")
    plt.ylabel("Latency (µs)")
    plt.legend(loc="best")

    _save(out_path, save_png)
    pstyle.report(out_path)


def plot_square(series: dict[str, dict[str, list[float]]], out_path: str,
                save_png: bool) -> None:
    """Notebook cell 16 layout: 7 x 4 in, log2 x axis (not in the paper; sized as if
    printed at \\columnwidth = 240 pt)."""
    pstyle = plot_fonts.paper_style(printed_width_pt=240.0, cropped_width_pt=443.5)
    plt.figure(figsize=(7, 4))
    x = np.array([32, 64, 128, 256, 512, 1024, 2048])  # bytes, for the log axis

    # NOTE: the original notebook cell drew the "Direct offload" markers at
    # the 25th percentile (plot_lists[4][3]) rather than the median.  The
    # two differ by <0.02 us for this series, so plotting the median is
    # visually identical.
    for label, (_, colour, marker) in SERIES.items():
        plt.plot(x, series[label]["median"], color=colour, marker=marker, label=label,
                 linewidth=pstyle.line, markersize=pstyle.marker)

    plt.xlabel("Data Size")
    plt.ylabel("Latency (µs)")
    plt.xscale("log", base=2)
    plt.xticks(x, X_LABELS)
    plt.legend(loc="best")

    _save(out_path, save_png)
    pstyle.report(out_path)


def _save(out_path: str, save_png: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"wrote {out_path}")
    if save_png:
        png_path = os.path.splitext(out_path)[0] + ".png"
        plt.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
        print(f"wrote {png_path}")
    plt.close()


# --------------------------------------------------------------------------- #
def print_table(series: dict[str, dict[str, list[float]]]) -> None:
    print(f"\nLatency (us) at {CLOCK_MHZ} MHz, {MAX_SAMPLES} samples per point")
    for label, stats in series.items():
        print(f"\n[{label}]  ({SERIES[label][0]})")
        print(f"{'size':>7} {'median':>8} {'p25':>8} {'p75':>8} {'p90':>8} {'p99.9':>8}")
        for i, size in enumerate(X_LABELS):
            print(f"{size:>7} {stats['median'][i]:>8.2f} {stats['p25'][i]:>8.2f} "
                  f"{stats['p75'][i]:>8.2f} {stats['p90'][i]:>8.2f} {stats['p99.9'][i]:>8.2f}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["wide", "square", "both"], default="wide",
                        help="figure layout to produce (default: wide)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"folder holding the raw .txt measurements (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", default=SCRIPT_DIR,
                        help="where to write the PDF(s) (default: next to this script)")
    parser.add_argument("--png", action="store_true",
                        help="also write a 200 dpi PNG next to each PDF")
    parser.add_argument("--quiet", action="store_true", help="do not print the summary table")
    args = parser.parse_args(argv)

    series = {label: load_series(args.data_dir, folder)
              for label, (folder, _, _) in SERIES.items()}
    if not args.quiet:
        print_table(series)

    if args.variant in ("wide", "both"):
        plot_wide(series, os.path.join(args.out_dir, "ClickNP_Result_no_shadow.pdf"), args.png)
    if args.variant in ("square", "both"):
        plot_square(series, os.path.join(args.out_dir, "ClickNP_Result_no_shadow_square.pdf"),
                    args.png)


if __name__ == "__main__":
    main()
