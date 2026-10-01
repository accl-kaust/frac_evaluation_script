# fRAC evaluation scripts

Plot scripts and raw measurements for the evaluation figures (Fig. 13–18).
Run every script from the repository root: each one reads its inputs from
`data/` and writes `<script name>.pdf` next to the script.

| Script | Data |
| --- | --- |
| `latency_throughput_fig_13.py` | `data/latency_throughput/` |
| `mixed_workload_fig_14.py` | `data/mixed_workload/` |
| `reassembly_fig_15.py` | `data/reassembly/` |
| `scalability_fig_16.py` | `data/scalability/` |
| `tail_latency_cdf_fig_17.py` | `data/scalability/` |
| `azure_trace_fig_18.py` | `data/azure_trace/` |

## Setup

```sh
pip install numpy pandas scipy matplotlib
```

The scalability traces (2.4 GB of per-thread latency samples, 1200 files) are
tracked compressed as `data/scalability/*.tar.xz`. Extract them once with

```sh
./unpack_data.sh
```

before running `tail_latency_cdf_fig_17.py`. `scalability_fig_16.py` reads the
committed `data/scalability/processed_latency_cache.txt` and only touches the
raw traces to fill in entries missing from that cache. The extracted
`data/scalability/top_k_*_inst/` directories are git-ignored.
