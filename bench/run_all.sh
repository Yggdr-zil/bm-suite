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
#   5-9. Benchmarks — GEMM, MemBW, VRAM, Interconnect, Inference
#  10. Score        — eCU scoring (v4 exponents, 8x H100 SXM ref)
#  11. Upload       — send results
#
# Environment variables:
#   RESULTS_DIR          — output directory (auto-created if not set)
#   CU_WARMUP_MIN_SECS  — minimum thermal soak time (default: 300 = 5 min)
#   CU_WARMUP_MAX_SECS  — max warmup time (default: 600 = 10 min)
#   CU_BENCH_MODEL       — model name/path for inference (optional)
#   CU_BENCH_MODEL_DIR   — directory containing model weights (default: /models)
#   CU_GEMM_DIM          — GEMM matrix dimension (default: 8192)
#   CU_GEMM_ITERS        — GEMM measured iterations (default: 200)
#   CU_GPU_CLOCK         — override: skip sustained clock detection, use this freq
#   CU_MEM_CLOCK         — override: lock mem clock to this freq
#   CU_WS_URL            — WebSocket URL for live telemetry streaming (optional)
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

# Helper: run a benchmark step, log failures but keep going
run_step() {
    local step_name="$1"
    shift
    if "$@"; then
        echo "  [OK] $step_name"
    else
        echo "  [FAIL] $step_name (exit code $?)"
        FAILED+=("$step_name")
    fi
}

# Helper: write a benchmark event file for the WS sender to pick up
emit_event() {
    local name="$1"
    local data="$2"
    local events_dir="${RUN_DIR}/.events"
    mkdir -p "$events_dir"
    echo "$data" > "${events_dir}/$(date -u +%Y%m%dT%H%M%S)_${name}"
}

# ─── [0/11] Preflight: Detect + Discover (no clock lock) ───
echo -e "\n[0/11] Preflight — detecting hardware..."
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
    # Kill WS sender
    [ -n "${WS_PID:-}" ] && kill $WS_PID 2>/dev/null && wait $WS_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

# ─── [1/11] Start Telemetry ───
echo -e "\n[1/11] Starting background telemetry..."
bash "${SCRIPT_DIR}/telemetry.sh" &
TELEMETRY_PID=$!
echo "  Telemetry PID: $TELEMETRY_PID"

# ─── [1.5] Start WebSocket streaming (optional) ───
WS_PID=""
WS_URL="${CU_WS_URL:-}"
if [ -n "$WS_URL" ]; then
    echo -e "\n[1.5] Starting WebSocket telemetry stream → ${WS_URL}..."
    python3 "${SCRIPT_DIR}/ws_sender.py" "$WS_URL" &
    WS_PID=$!
    echo "  WS sender PID: $WS_PID"
else
    echo -e "\n[1.5] WebSocket streaming — skipped (no CU_WS_URL set)"
fi

# ─── [2/11] Thermal Soak (5 min minimum) ───
echo -e "\n[2/11] Thermal soak — sustained GEMM for thermal equilibrium (5 min minimum)..."
run_step "thermal_warmup" python3 "${SCRIPT_DIR}/thermal_gate.py"

# ─── [3/11] Sustained Clock Detection (on HOT GPU) ───
# Skip if user explicitly set CU_GPU_CLOCK or clocks aren't deprecated
if [ -n "${CU_GPU_CLOCK:-}" ]; then
    echo -e "\n[3/11] Sustained clock — skipped (CU_GPU_CLOCK=${CU_GPU_CLOCK} set by user)"
elif [ "${CLOCKS_DEPRECATED:-False}" = "True" ]; then
    echo -e "\n[3/11] Sustained clock — detecting on thermally-saturated GPU..."
    run_step "sustained_clock" python3 "${SCRIPT_DIR}/find_sustained_clock.py"
else
    echo -e "\n[3/11] Sustained clock — skipped (default applications clocks available)"
fi

# ─── [4/11] Lock Clocks ───
echo -e "\n[4/11] Locking clocks..."
run_step "lock_clocks" python3 "${SCRIPT_DIR}/lock_clocks.py"

# ─── [5/11] GEMM Benchmark ───
echo -e "\n[5/11] Benchmark: GEMM (FP throughput)..."
run_step "gemm" python3 "${SCRIPT_DIR}/gemm.py"
emit_event "gemm_done" "GEMM benchmark complete"

# ─── [6/11] Memory Bandwidth ───
echo -e "\n[6/11] Benchmark: Memory bandwidth..."
run_step "membw" python3 "${SCRIPT_DIR}/membw.py"
emit_event "membw_done" "Memory bandwidth benchmark complete"

# ─── [7/11] VRAM Capacity ───
echo -e "\n[7/11] Benchmark: VRAM capacity..."
run_step "vram" python3 "${SCRIPT_DIR}/vram.py"
emit_event "vram_done" "VRAM benchmark complete"

# ─── [8/11] Interconnect ───
echo -e "\n[8/11] Benchmark: Interconnect bandwidth..."
run_step "interconnect" bash "${SCRIPT_DIR}/interconnect.sh"
emit_event "interconnect_done" "Interconnect benchmark complete"

# ─── [9/11] Inference ───
echo -e "\n[9/11] Benchmark: Inference throughput (vLLM)..."
run_step "inference" python3 "${SCRIPT_DIR}/inference.py"
emit_event "inference_done" "Inference benchmark complete"

# ─── Compile Report ───
echo -e "\nCompiling final report..."
run_step "report" python3 "${SCRIPT_DIR}/report.py"

# ─── [10/11] eCU Scoring ───
echo -e "\nScoring with eCU (v4 exponents, 8x H100 SXM reference)..."
run_step "ecu_score" python3 "${SCRIPT_DIR}/score.py" "$RUN_DIR/benchmark_report.json"
emit_event "scored" "eCU scoring complete"

# ─── [11/11] Upload Results ───
echo -e "\nUploading results..."
run_step "upload" bash "${SCRIPT_DIR}/upload.sh"

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
# Show eCU scores if available
python3 -c "
import json, sys
try:
    r = json.load(open('${RUN_DIR}/benchmark_report.json'))
    s = r.get('ecu_scores', {})
    if s:
        print(f'  eTCU={s.get(\"eTCU\", s.get(\"etcu\", \"?\"))}  eICU={s.get(\"eICU\", s.get(\"eicu\", \"?\"))}  eCU={s.get(\"eCU\", s.get(\"ecu\", \"?\"))}')
        print(f'  Reference: {s.get(\"reference_gpu\", \"?\")}')
except: pass
" 2>/dev/null || true
echo "================================================================"
