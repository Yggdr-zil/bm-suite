#!/bin/bash
# CU Benchmark Suite — Preflight: Detect, Discover, Lock
# Discovers hardware parameters dynamically and locks for deterministic benchmarking.
# NOTE: Do NOT use set -e here — this script is meant to be sourced,
# and many nvidia-smi lock commands fail on consumer GPUs (expected).
set -u

RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/results/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR
ENV_FILE="${RESULTS_DIR}/00_environment.json"

echo "Results: $RESULTS_DIR"

# ─── Platform Detection ───
PLATFORM=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    PLATFORM="nvidia"
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null; then
    PLATFORM="amd"
else
    echo "FATAL: No supported accelerator detected (need nvidia-smi or rocm-smi)"
    exit 1
fi
echo "Platform: $PLATFORM"
export PLATFORM

# ─── Common helper ───
json_escape() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))"; }

if [ "$PLATFORM" = "nvidia" ]; then
    # ─── NVIDIA: Discover ───
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | xargs)
    ECC_MODE=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "N/A")

    # Base clocks — priority: env var > default_applications > SKIP (never max boost)
    # User can set CU_GPU_CLOCK / CU_MEM_CLOCK from find_sustained_clock.py results
    BASE_GPU_CLK="${CU_GPU_CLOCK:-}"
    BASE_MEM_CLK="${CU_MEM_CLOCK:-}"
    CLOCKS_DEPRECATED=false

    if [ -z "$BASE_GPU_CLK" ]; then
        BASE_GPU_CLK=$(nvidia-smi --query-gpu=clocks.default_applications.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "")
        if [ -z "$BASE_GPU_CLK" ] || echo "$BASE_GPU_CLK" | grep -qi "deprecated\|error\|N/A"; then
            CLOCKS_DEPRECATED=true
            BASE_GPU_CLK=""
        fi
    fi
    if [ -z "$BASE_MEM_CLK" ]; then
        BASE_MEM_CLK=$(nvidia-smi --query-gpu=clocks.default_applications.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "")
        if [ -z "$BASE_MEM_CLK" ] || echo "$BASE_MEM_CLK" | grep -qi "deprecated\|error\|N/A"; then
            BASE_MEM_CLK=""
        fi
    fi

    MAX_GPU_CLK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
    MAX_MEM_CLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")

    if $CLOCKS_DEPRECATED && [ -z "$BASE_GPU_CLK" ]; then
        echo "NOTE: Default Applications Clocks deprecated on this GPU."
        echo "  Max boost: ${MAX_GPU_CLK} MHz (NOT locking to this — would cause throttling)"
        echo "  Run 'python3 bench/find_sustained_clock.py' to find sustained clock,"
        echo "  then set: export CU_GPU_CLOCK=<freq>  before running the suite."
        echo "  Proceeding WITHOUT clock lock — results will have natural clock variance."
    fi

    DEFAULT_PL=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | xargs)
    PL_INT=${DEFAULT_PL%.*}  # strip decimals

    echo "GPU: $GPU_MODEL x$GPU_COUNT | Compute: $COMPUTE_CAP | Driver: $DRIVER"
    echo "Clocks: GPU=${BASE_GPU_CLK:-unlocked} Mem=${BASE_MEM_CLK:-unlocked} (max: ${MAX_GPU_CLK}/${MAX_MEM_CLK}) | TDP: ${PL_INT}W | ECC: $ECC_MODE"

    # ─── NVIDIA: Check for clean GPU ───
    USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | xargs)
    USED_MEM_INT=${USED_MEM%.*}
    PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l || echo 0)
    if [ "$USED_MEM_INT" -gt 500 ] || [ "$PROCS" -gt 0 ]; then
        echo "WARNING: GPU not clean (${USED_MEM_INT} MiB used, ${PROCS} processes)"
        echo "For accurate benchmarks, ensure no other GPU workloads are running."
        echo "Continuing anyway — results may be affected."
    fi

    # ─── NVIDIA: Lock ───
    echo "Locking hardware..."

    # Persistence mode
    nvidia-smi -pm 1 2>/dev/null || echo "WARNING: Could not set persistence mode (need root?)"

    # GPU clock lock — only if we have a valid clock target
    if [ -n "$BASE_GPU_CLK" ]; then
        nvidia-smi -lgc ${BASE_GPU_CLK},${BASE_GPU_CLK} 2>/dev/null && \
            echo "  GPU clock locked: ${BASE_GPU_CLK} MHz" || \
            echo "  WARNING: Could not lock GPU clock (need root/admin?)"
    else
        echo "  GPU clock: UNLOCKED (no target frequency — set CU_GPU_CLOCK)"
    fi

    # Memory clock lock — only if we have a valid target
    if [ -n "$BASE_MEM_CLK" ]; then
        if [[ "$COMPUTE_CAP" == 9.* ]]; then
            nvidia-smi --lock-memory-clocks-deferred ${BASE_MEM_CLK},${BASE_MEM_CLK} 2>/dev/null && \
                echo "  Mem clock locked (deferred): ${BASE_MEM_CLK} MHz" || \
                echo "  WARNING: Could not lock memory clocks (Hopper deferred mode)"
        else
            nvidia-smi -lmc ${BASE_MEM_CLK},${BASE_MEM_CLK} 2>/dev/null && \
                echo "  Mem clock locked: ${BASE_MEM_CLK} MHz" || \
                echo "  WARNING: Could not lock memory clocks"
        fi
    else
        echo "  Mem clock: UNLOCKED (no target frequency — set CU_MEM_CLOCK)"
    fi

    # Power limit
    nvidia-smi -pl $PL_INT 2>/dev/null && \
        echo "  Power limit set: ${PL_INT}W" || \
        echo "  WARNING: Could not set power limit"

    # Record starting temperature
    START_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1 | xargs)

    # Topology
    TOPO=$(nvidia-smi topo -m 2>/dev/null || echo "unavailable")

    # Serial numbers
    SERIALS=$(nvidia-smi --query-gpu=serial --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//' || echo "unavailable")

elif [ "$PLATFORM" = "amd" ]; then
    # ─── AMD: Discover ───
    HAS_AMDSMI=false
    command -v amd-smi &>/dev/null && HAS_AMDSMI=true

    if $HAS_AMDSMI; then
        GPU_MODEL=$(amd-smi static --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0].get('asic',{}).get('market_name','unknown'))" 2>/dev/null || echo "unknown")
    else
        GPU_MODEL=$(rocm-smi --showproductname 2>/dev/null | grep -oP '(?<=Card series:\s{8}).*' | head -1 | xargs || echo "unknown")
    fi

    GPU_COUNT=$(python3 -c "
try:
    import torch
    print(torch.cuda.device_count())
except:
    import subprocess
    out = subprocess.check_output(['rocm-smi', '--showcount'], text=True)
    import re
    m = re.search(r'(\d+)', out)
    print(m.group(1) if m else 1)
")
    DRIVER=$(rocm-smi --showdriverversion 2>/dev/null | grep -oP '[\d.]+' | head -1 || echo "unknown")
    COMPUTE_CAP="N/A"
    ECC_MODE=$(rocm-smi --showras 2>/dev/null | head -5 || echo "N/A")

    echo "GPU: $GPU_MODEL x$GPU_COUNT | Driver: $DRIVER"

    # ─── AMD: Check for clean GPU ───
    if $HAS_AMDSMI; then
        PROCS=$(amd-smi process --json 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    count=sum(len(gpu.get('process_list',[])) for gpu in d)
    print(count)
except: print(0)" 2>/dev/null || echo 0)
    else
        PROCS=$(rocm-smi --showpids 2>/dev/null | grep -c "^\s*[0-9]" || echo 0)
    fi
    if [ "$PROCS" -gt 0 ]; then
        echo "WARNING: $PROCS processes detected on GPU. Results may be affected."
    fi

    # ─── AMD: Lock ───
    echo "Locking hardware..."

    rocm-smi --setperflevel high 2>/dev/null && \
        echo "  Perf level: high" || \
        echo "  WARNING: Could not set perf level"

    rocm-smi --setperfdeterminism 1900 2>/dev/null && \
        echo "  Determinism mode: 1900 MHz ceiling" || \
        echo "  WARNING: Could not set perf determinism"

    if $HAS_AMDSMI; then
        amd-smi set -o 750 2>/dev/null && echo "  Power limit set: 750W" || \
            echo "  WARNING: Could not set power limit via amd-smi"
    else
        rocm-smi --setpoweroverdrive 750 2>/dev/null && echo "  Power limit set: 750W" || \
            echo "  WARNING: Could not set power limit"
    fi

    BASE_GPU_CLK="1900"
    BASE_MEM_CLK="N/A"
    PL_INT="750"

    if $HAS_AMDSMI; then
        START_TEMP=$(amd-smi metric -t --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0].get('temperature',{}).get('hotspot',0))" 2>/dev/null || echo "0")
    else
        START_TEMP=$(rocm-smi --showtemp 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "0")
    fi

    TOPO=$(rocm-smi --showtopo 2>/dev/null || echo "unavailable")
    SERIALS="N/A"
fi

# ─── Write environment JSON ───
python3 << PYEOF
import json, platform as plat, os

env = {
    "platform": "$PLATFORM",
    "gpu_model": "$GPU_MODEL",
    "gpu_count": int("$GPU_COUNT"),
    "driver_version": "$DRIVER",
    "compute_capability": "$COMPUTE_CAP",
    "ecc_mode": "$ECC_MODE",
    "gpu_clock_locked_mhz": "${BASE_GPU_CLK:-unlocked}",
    "gpu_clock_max_mhz": "${MAX_GPU_CLK:-unknown}",
    "mem_clock_locked_mhz": "${BASE_MEM_CLK:-unlocked}",
    "mem_clock_max_mhz": "${MAX_MEM_CLK:-unknown}",
    "power_limit_watts": int("${PL_INT%.*}" or "0"),
    "start_temp_c": float("${START_TEMP}" or "0"),
    "gpu_serials": "$SERIALS",
    "host_kernel": plat.release(),
    "host_os": plat.platform(),
    "python_version": plat.python_version(),
    "pytorch_version": "",
    "cuda_runtime": "",
}

try:
    import torch
    env["pytorch_version"] = torch.__version__
    env["cuda_runtime"] = torch.version.cuda or "N/A"
except ImportError:
    pass

with open("$ENV_FILE", "w") as f:
    json.dump(env, f, indent=2)

print(f"Environment saved: $ENV_FILE")
PYEOF

echo "Preflight complete. Start temp: ${START_TEMP}C"
export GPU_COUNT
export GPU_MODEL
