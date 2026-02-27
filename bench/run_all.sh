#!/bin/bash
# CU Benchmark Suite v1.0 — Main Orchestrator
# Usage: bash run_all.sh
#
# Step order (critical for measurement accuracy):
#   0. Preflight    — detect hardware, set fans, persistence mode (NO clock lock)
#   1. Telemetry    — background 1Hz GPU metrics
#   2. Thermal soak — 5 min minimum sustained GEMM until true steady state
#   3. Sustained clock — find real sustained frequency on HOT GPU
#   4. Lock clocks  — lock to sustained frequency
#   5-8. Benchmarks — GEMM, MemBW, VRAM, Interconnect, Inference
#   9. Report       — compile final report
#
# Environment variables:
#   RESULTS_DIR          — output directory (auto-created if not set)
#   CU_WARMUP_MIN_SECS  — minimum thermal soak time (default: 300 = 5 min)
#   CU_WARMUP_MAX_SECS  — max warmup time (default: 600 = 10 min)
#   CU_BENCH_MODEL       — model name/path for inference (optional)
#   CU_BENCH_MODEL_DIR   — directory containing model weights (default: /models)
#   CU_GEMM_DIM          — GEMM matrix dimension (default: 8192)
#   CU_GEMM_ITERS        — GEMM measured iterations (default: 10000)
#   CU_MEMBW_ITERS       — membw measured iterations (default: 5000)
#   CU_QUICK             — set to 1 for fast mode (1000/500 iters)
#   CU_GPU_CLOCK         — override: skip sustained clock detection, use this freq
#   CU_MEM_CLOCK         — override: lock mem clock to this freq
set -uo pipefail
# NOTE: no -e — we handle errors per-benchmark so one failure doesn't kill the suite

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BM_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
FAILED=()

# Results directory: explicit $RESULTS_DIR, or auto-create under bm-suite/results/
if [ -n "${RESULTS_DIR:-}" ]; then
    RUN_DIR="${RESULTS_DIR}"
else
    RUN_DIR="${BM_ROOT}/results/run_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"
export RESULTS_DIR="$RUN_DIR"

echo "================================================================"
echo "  CU Benchmark Suite v1.0"
echo "  $(date -u)"
echo "  Output: $RUN_DIR"
echo "================================================================"

# ─── Stage events CSV ───
STAGE_EVENTS="${RUN_DIR}/stage_events.csv"
echo "timestamp_utc,stage_name,event,exit_code" > "$STAGE_EVENTS"

log_stage_event() {
    # Usage: log_stage_event <stage_name> <start|end> [exit_code]
    local ts exit_code="${3:-0}"
    ts=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
    echo "${ts},${1},${2},${exit_code}" >> "$STAGE_EVENTS"
}

# Helper: run a benchmark step, log failures but keep going
run_step() {
    local step_name="$1"
    shift
    log_stage_event "$step_name" "start"
    if "$@"; then
        log_stage_event "$step_name" "end" "0"
        echo "  [OK] $step_name"
    else
        local ec=$?
        log_stage_event "$step_name" "end" "$ec"
        echo "  [FAIL] $step_name (exit code $ec)"
        FAILED+=("$step_name")
    fi
}

# ─── [0/9] Preflight: Detect + Discover (no clock lock) ───
echo -e "\n[0/9] Preflight — detecting hardware..."
python3 "${SCRIPT_DIR}/preflight.py"
# Source platform vars for telemetry.sh and cleanup
source "${RESULTS_DIR}/_platform.sh"

# ─── Cleanup on exit/cancel (Ctrl+C) ───
cleanup() {
    echo -e "\nCleaning up..."
    # Kill telemetry
    [ -n "${TELEMETRY_PID:-}" ] && kill $TELEMETRY_PID 2>/dev/null && wait $TELEMETRY_PID 2>/dev/null
    # Unlock clocks
    if [ "${PLATFORM:-nvidia}" = "nvidia" ]; then
        nvidia-smi -rgc >/dev/null 2>&1 || true
        nvidia-smi -rmc >/dev/null 2>&1 || true
        echo "GPU clocks unlocked."
    else
        rocm-smi -r 2>/dev/null || true
        echo "GPU reset."
    fi
}
trap cleanup EXIT INT TERM

# ─── [1/9] Start Telemetry ───
echo -e "\n[1/9] Starting background telemetry..."
bash "${SCRIPT_DIR}/telemetry.sh" &
TELEMETRY_PID=$!
echo "  Telemetry PID: $TELEMETRY_PID"

# ─── [2/9] Thermal Soak (5 min minimum) ───
echo -e "\n[2/9] Thermal soak — sustained GEMM for thermal equilibrium (5 min minimum)..."
run_step "thermal_warmup" python3 "${SCRIPT_DIR}/thermal_gate.py"

# ─── [3/9] Sustained Clock Detection (on HOT GPU) ───
# Skip if user explicitly set CU_GPU_CLOCK or clocks aren't deprecated
if [ -n "${CU_GPU_CLOCK:-}" ]; then
    echo -e "\n[3/9] Sustained clock — skipped (CU_GPU_CLOCK=${CU_GPU_CLOCK} set by user)"
elif [ "${CLOCKS_DEPRECATED:-False}" = "True" ]; then
    echo -e "\n[3/9] Sustained clock — detecting on thermally-saturated GPU..."
    run_step "sustained_clock" python3 "${SCRIPT_DIR}/find_sustained_clock.py"
else
    echo -e "\n[3/9] Sustained clock — skipped (default applications clocks available)"
fi

# ─── [4/9] Lock Clocks ───
echo -e "\n[4/9] Locking clocks..."
run_step "lock_clocks" python3 "${SCRIPT_DIR}/lock_clocks.py"

# ─── [5/9] GEMM Benchmark ───
echo -e "\n[5/9] Benchmark: GEMM (FP throughput)..."
run_step "gemm" python3 "${SCRIPT_DIR}/gemm.py"

# ─── [6/9] Memory Bandwidth ───
echo -e "\n[6/9] Benchmark: Memory bandwidth..."
run_step "membw" python3 "${SCRIPT_DIR}/membw.py"

# ─── [7/9] VRAM Capacity ───
echo -e "\n[7/9] Benchmark: VRAM capacity..."
run_step "vram" python3 "${SCRIPT_DIR}/vram.py"

# ─── [8/9] Interconnect ───
echo -e "\n[8/9] Benchmark: Interconnect bandwidth..."
run_step "interconnect" bash "${SCRIPT_DIR}/interconnect.sh"

# ─── [9/9] Inference ───
echo -e "\n[9/9] Benchmark: Inference throughput (vLLM)..."
run_step "inference" python3 "${SCRIPT_DIR}/inference.py"

# ─── Compile Report ───
echo -e "\nCompiling final report..."
run_step "report" python3 "${SCRIPT_DIR}/report.py"

# ─── Auto-plot (PNG + HTML) ───
echo -e "\nGenerating plots..."
run_step "plot" python3 "${SCRIPT_DIR}/plot.py"

# ─── Upload results (before terminal pause, so data is safe first) ───
if [ -n "${CU_UPLOAD_DEST:-}" ] || [ -n "${CU_UPLOAD_WEBHOOK:-}" ]; then
    bash "${SCRIPT_DIR}/upload.sh"
fi

# ─── Terminal visualization (pauses for 'q' — data already sent) ───
echo -e "\nRendering terminal charts..."
python3 "${SCRIPT_DIR}/plot_terminal.py" || true

# ─── Fix ownership if run with sudo ───
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER:$SUDO_USER" "$RUN_DIR" 2>/dev/null || true
    echo "Ownership set to $SUDO_USER"
fi

echo -e "\n================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  COMPLETE — all benchmarks passed"
else
    echo "  COMPLETE — ${#FAILED[@]} benchmark(s) failed: ${FAILED[*]}"
fi
echo "  Results: $RUN_DIR/benchmark_report.json"
echo "  Plots:   $RUN_DIR/plots/summary.html"
if [ -n "${CU_UPLOAD_DEST:-}" ]; then
    echo "  Upload:  ${CU_UPLOAD_DEST}"
fi
echo "================================================================"
