#!/bin/bash
# CU Benchmark Suite — Background Telemetry Recorder
# Logs GPU temp, power, clocks every second to CSV.
# Run in background: bash telemetry.sh &
# Kill when done: kill $!
set -u

PLATFORM="${PLATFORM:-nvidia}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/results}"
mkdir -p "$RESULTS_DIR"
OUT_FILE="${RESULTS_DIR}/telemetry.csv"

if [ "$PLATFORM" = "nvidia" ]; then
    echo "timestamp,gpu_idx,temp_c,temp_mem_c,power_w,clock_gpu_mhz,clock_mem_mhz,util_gpu_pct,mem_used_mib,throttle_reasons" > "$OUT_FILE"
    nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,power.draw,clocks.gr,clocks.mem,utilization.gpu,memory.used,throttle_reasons \
        --format=csv,noheader,nounits -l 1 >> "$OUT_FILE"

elif [ "$PLATFORM" = "amd" ]; then
    echo "timestamp,gpu_idx,temp_hotspot_c,temp_mem_c,power_w,clock_sclk_mhz,clock_mclk_mhz" > "$OUT_FILE"

    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    HAS_AMDSMI=false
    command -v amd-smi &>/dev/null && HAS_AMDSMI=true

    while true; do
        TS=$(date -u +"%Y/%m/%d %H:%M:%S.%3N")

        if $HAS_AMDSMI; then
            METRICS=$(amd-smi metric -t -p -c --json 2>/dev/null || echo "[]")
            python3 -c "
import json, sys
ts = '$TS'
try:
    data = json.loads('''$METRICS''')
    for i, gpu in enumerate(data):
        hotspot = gpu.get('temperature', {}).get('hotspot', 0)
        mem_temp = gpu.get('temperature', {}).get('mem', 0)
        power = gpu.get('power', {}).get('socket_power', 0)
        sclk = gpu.get('clock', {}).get('sclk', 0)
        mclk = gpu.get('clock', {}).get('mclk', 0)
        print(f'{ts}, {i}, {hotspot}, {mem_temp}, {power}, {sclk}, {mclk}')
except: pass
" >> "$OUT_FILE"
        else
            for i in $(seq 0 $((GPU_COUNT-1))); do
                TEMP=$(rocm-smi -d $i --showtemp 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "0")
                echo "$TS, $i, $TEMP, 0, 0, 0, 0" >> "$OUT_FILE"
            done
        fi

        sleep 1
    done
else
    echo "Unknown platform: $PLATFORM"
    exit 1
fi
