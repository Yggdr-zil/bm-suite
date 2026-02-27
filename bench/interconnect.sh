#!/bin/bash
# CU Benchmark Suite — Interconnect Bandwidth Benchmark
# Uses NCCL (NVIDIA) or RCCL (AMD) all_reduce_perf to measure
# inter-GPU bandwidth (NVLink / Infinity Fabric).
set -euo pipefail

PLATFORM="${PLATFORM:-nvidia}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/results}"
mkdir -p "$RESULTS_DIR"
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

echo "Interconnect Benchmark | ${PLATFORM} | ${GPU_COUNT} GPU(s)"

# ─── Single GPU: skip ───
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  Single GPU detected — no interconnect to measure. Skipping."
    python3 -c "
import json, os
with open(os.path.join('${RESULTS_DIR}', '04_interconnect.json'), 'w') as f:
    json.dump({'skipped': True, 'reason': 'single_gpu', 'gpu_count': 1}, f, indent=2)
"
    exit 0
fi

# ─── Find the benchmark binary ───
BENCH_BIN=""
SEARCH_PATHS=(
    "/opt/nccl-tests/build/all_reduce_perf"
    "/bench/nccl-tests/build/all_reduce_perf"
    "/opt/rccl-tests/build/all_reduce_perf"
    "/bench/rccl-tests/build/all_reduce_perf"
)

for path in "${SEARCH_PATHS[@]}"; do
    if [ -x "$path" ]; then
        BENCH_BIN="$path"
        break
    fi
done

if [ -z "$BENCH_BIN" ]; then
    echo "  WARNING: nccl-tests/rccl-tests binary not found. Skipping."
    echo "  Searched: ${SEARCH_PATHS[*]}"
    python3 -c "
import json, os
with open(os.path.join('${RESULTS_DIR}', '04_interconnect.json'), 'w') as f:
    json.dump({'skipped': True, 'reason': 'binary_not_found', 'gpu_count': ${GPU_COUNT}}, f, indent=2)
"
    exit 0
fi

echo "  Binary: $BENCH_BIN"
echo "  Running all_reduce sweep: 1MB -> 2GB, ${GPU_COUNT} GPUs, 50 iters..."

RAW_FILE="${RESULTS_DIR}/04_interconnect_raw.txt"

$BENCH_BIN \
    -b 1M -e 2G -f 2 \
    -g $GPU_COUNT \
    -n 50 \
    -w 10 \
    2>&1 | tee "$RAW_FILE"

# ─── Parse results ───
python3 << 'PYEOF'
import json
import re
import os

raw_file = os.environ["RESULTS_DIR"] + "/04_interconnect_raw.txt"
out_file = os.environ["RESULTS_DIR"] + "/04_interconnect.json"
gpu_count = int(os.environ.get("GPU_COUNT", "1"))

results = []
with open(raw_file) as f:
    for line in f:
        line = line.strip()
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            size = int(parts[0])
            algbw = float(parts[7])
            busbw = float(parts[8])
            results.append({
                "size_bytes": size,
                "algbw_gbps": algbw,
                "busbw_gbps": busbw,
            })
        except (ValueError, IndexError):
            continue

if not results:
    output = {
        "skipped": False,
        "error": "no parseable results",
        "gpu_count": gpu_count,
    }
else:
    peak = max(results, key=lambda r: r["busbw_gbps"])
    gb1 = 1073741824
    closest_1g = min(results, key=lambda r: abs(r["size_bytes"] - gb1))

    output = {
        "skipped": False,
        "gpu_count": gpu_count,
        "peak_busbw_gbps": peak["busbw_gbps"],
        "peak_at_size_mb": round(peak["size_bytes"] / 1e6, 1),
        "busbw_at_1gb_gbps": closest_1g["busbw_gbps"],
        "all_results": results,
    }
    print(f"\n  Peak bus BW: {output['peak_busbw_gbps']} GB/s at {output['peak_at_size_mb']}MB")
    print(f"  Bus BW @ 1GB: {output['busbw_at_1gb_gbps']} GB/s")

with open(out_file, "w") as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {out_file}")
PYEOF
