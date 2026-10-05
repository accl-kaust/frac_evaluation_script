# Trace replay with tperf

## Function IDs (`app` column in processed_trace.csv)

| app | function |
|-----|----------|
| 1   | TOPK     |
| 2   | CNN      |
| 3   | LOGIT    |
| 5   | NORM     |

## How `sleep_time` is handled in libtpa

`sleep_time` is the gap in seconds between the start of this request and the start of the next one.
It was computed in `preprocess_trace.py` as `start_timestamp[i] - start_timestamp[i-1]` from the raw trace.

In `app/tperf/client.c` the client replays rows in order on one connection:

1. Send row `i` with `func`, `request_size`, `response_size` from the CSV.
2. Record `next_send_ns = now + sleep_time[i] * 1e9`.
3. Wait until the response for row `i` has been fully read **and** `now >= next_send_ns`.
4. Send row `i+1`.

So `sleep_time` is a minimum inter-send gap, not a pause after the response.

## Run

```bash
# server
sudo TPA_ID=server TPA_ETH_DEV=enp195s0f1np1 tpa run build/bin/app/tperf -s -n 1 -S 1 -p 3000

# client
sudo -E TPA_ID=client TPA_ETH_DEV=enp194s0f1np1 TPA_CFG="tcp {tso = 0; }" \
  tpa run build/bin/app/tperf -c 172.24.5.50 -n 1 -t rr -Z 0 -S 0 -p 3000 -m 1024 \
  -E /home/yangz0e/trace_libtpa/libtpa/processed_trace.csv
```

Add `-L 1 -D trace_log` to log per-request send/response timestamps and latency.
