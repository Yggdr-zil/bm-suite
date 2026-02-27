#!/bin/bash
# CU Benchmark Suite — Local runner (no Docker)
# Runs benchmarks using your system's PyTorch installation.
#
# Usage:
#   cd bm-suite
#   bash run_local.sh
#
# Requirements:
#   - Python 3.8+
#   - PyTorch with CUDA support
#   - rich (pip install rich)
#   - nvidia-smi or rocm-smi
#
# Optional:
#   - vLLM (for inference benchmark)
#   - nccl-tests compiled (for interconnect benchmark)
#
# Environment variables:
#   RESULTS_DIR          — output directory (auto-created if not set)
#   CU_BENCH_MODEL       — model name or path for inference benchmark
#   CU_WARMUP_MIN_SECS   — minimum thermal soak time (default: 300 = 5 min)
#   CU_WARMUP_MAX_SECS   — max thermal warmup time (default: 600 = 10 min)
#   CU_GEMM_DIM          — GEMM matrix size (default: 8192, use 4096 for small GPUs)
#   CU_GEMM_ITERS        — GEMM iterations (default: 200)
#   CU_GPU_CLOCK         — override: skip sustained clock detection, use this freq
#   CU_MEM_CLOCK         — override: lock mem clock to this freq
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/bench" && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/run_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RESULTS_DIR"

echo "CU Benchmark Suite — Local Mode"
echo "Results: $RESULTS_DIR"
echo ""

# Quick sanity check
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || {
    echo "ERROR: PyTorch CUDA not available. Need a CUDA-capable GPU and PyTorch with CUDA."
    exit 1
}

exec bash "${SCRIPT_DIR}/run_all.sh"
