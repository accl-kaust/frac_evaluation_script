#!/bin/sh
# Extract the compressed scalability traces (data/scalability/*.tar.xz) into
# data/scalability/top_k_*_inst/, where scalability_fig_16.py and
# tail_latency_cdf_fig_17.py expect them.
set -e
cd "$(dirname "$0")/data/scalability"
for f in *.tar.xz; do
    echo "extracting $f"
    tar xf "$f"
done
