#!/bin/bash
# CU Benchmark Suite — Parallel Multi-GPU Runner
#
# Runs run_all.sh simultaneously on each GPU (one process per GPU).
# All GPU results land in per-GPU subdirectories and upload independently.
#
# Usage:
#   bash run_parallel.sh              # auto-detect all GPUs
#   NUM_GPUS=4 bash run_parallel.sh   # limit to first 4 GPUs
#   CU_QUICK=1 bash run_parallel.sh   # quick mode on all GPUs
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

# Detect GPU count
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}"
if [ "${NUM_GPUS:-0}" -eq 0 ]; then
    echo "ERROR: No NVIDIA GPUs detected (is nvidia-smi available?)"
    exit 1
fi

BASE_RESULTS="${RESULTS_DIR:-$(dirname "$SCRIPT_DIR")/results}"
RUN_BASE="${BASE_RESULTS}/run_${TIMESTAMP}"
mkdir -p "$RUN_BASE"

echo "================================================================"
echo "  CU Benchmark Suite — Parallel Mode"
echo "  $(date -u)"
echo "  GPUs detected: $NUM_GPUS"
echo "  Output: $RUN_BASE"
echo "================================================================"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_DIR="${RUN_BASE}/gpu${i}"
    mkdir -p "$GPU_DIR"
    echo "  Launching GPU $i → $GPU_DIR"
    CUDA_VISIBLE_DEVICES=$i \
    RESULTS_DIR="$GPU_DIR" \
        bash "${SCRIPT_DIR}/run_all.sh" \
        > "${RUN_BASE}/gpu${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NUM_GPUS GPU benchmark processes launched in parallel."
echo "Tailing GPU 0 — Ctrl+C stops the tail, benchmarks keep running."
echo "────────────────────────────────────────────────────────────────"
tail -f "${RUN_BASE}/gpu0.log" &
TAIL_PID=$!

# Wait for all GPU benchmarks
ALL_OK=true
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  [OK] GPU $i complete"
    else
        echo "  [FAIL] GPU $i — see ${RUN_BASE}/gpu${i}.log"
        ALL_OK=false
    fi
done

kill "$TAIL_PID" 2>/dev/null || true

echo ""
echo "================================================================"
if $ALL_OK; then
    echo "  ALL $NUM_GPUS GPUs COMPLETE"
else
    echo "  COMPLETE WITH ERRORS — check per-GPU logs in $RUN_BASE/"
fi
echo "  Results: $RUN_BASE/gpu*/benchmark_report.json"
echo "================================================================"
