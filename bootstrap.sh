#!/bin/bash
# CU Benchmark Suite — Bootstrap
#
# Installs cu-bench on any GPU instance with CUDA + Python already present
# (Lambda Labs, CoreWeave, RunPod, Vast.ai, AWS Deep Learning AMI, etc.)
#
# ─── Usage ────────────────────────────────────────────────────────────────────
#
#   Minimal (results stay on instance):
#     bash <(curl -fsSL https://raw.githubusercontent.com/Yggdr-zil/bm-suite/main/bootstrap.sh)
#
#   With upload (results auto-ship to your server when done):
#     CU_UPLOAD_DEST=user@yourserver:/data/cu-inbox/ \
#     CU_UPLOAD_KEY=$(base64 -w0 ~/.ssh/id_bench) \
#     bash <(curl -fsSL https://raw.githubusercontent.com/Yggdr-zil/bm-suite/main/bootstrap.sh)
#
#   From local machine (pipe over SSH — credentials never touch the remote):
#     CU_UPLOAD_DEST=user@yourserver:/data/cu-inbox/ \
#     CU_UPLOAD_KEY=$(base64 -w0 ~/.ssh/id_bench) \
#     ssh ubuntu@<instance-ip> 'bash -s' < ~/projects/cm/bm-suite/bootstrap.sh
#
# ─── After bootstrap ──────────────────────────────────────────────────────────
#   bench-run              # full suite (~45 min on H100)
#   CU_QUICK=1 bench-run   # quick validation (~5 min)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-${HOME}/cu-bench}"
REPO_URL="https://github.com/Yggdr-zil/bm-suite.git"

echo ""
echo "================================================================"
echo "  CU Benchmark Suite — Bootstrap"
echo "  Installing to: ${INSTALL_DIR}"
echo "================================================================"
echo ""

# ── Clone or update bm-suite (public repo — no auth needed) ──────────────────
if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "  Already installed — pulling latest..."
    cd "$INSTALL_DIR"
    git pull --ff-only
else
    echo "  Cloning bm-suite..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# ── Run setup (installs deps + creates bench-run) ─────────────────────────────
echo ""
chmod +x setup
bash setup

echo ""
echo "================================================================"
echo "  Bootstrap complete."
echo ""
echo "  Run benchmarks:"
echo "    bench-run              # full suite (~45 min on H100)"
echo "    CU_QUICK=1 bench-run   # quick validation (~5 min)"
echo "================================================================"
echo ""
