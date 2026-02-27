#!/bin/bash
# CU Benchmark Suite — Docker convenience runner
# All required flags are baked in. Configure via .env file.
#
# ─── Setup (one time per machine) ────────────────────────────────────────────
#   cp .env.example .env
#   nano .env          # set CU_UPLOAD_DEST and CU_IMAGE
#   ./run.sh           # that's it, forever
#
# ─── Override at runtime ─────────────────────────────────────────────────────
#   CU_QUICK=1 ./run.sh                        # fast validation (~5 min)
#   ./run.sh -v /path/to/models:/models        # add inference benchmark
#   CU_UPLOAD_DEST=user@host:/path ./run.sh    # override upload destination
#
# ─── SSH key for upload ───────────────────────────────────────────────────────
#   Option A (recommended): set SSH_KEY_PATH in .env
#     SSH_KEY_PATH=~/.ssh/id_ed25519
#   Option B: base64 env var (no file mount needed, works anywhere)
#     CU_UPLOAD_KEY=$(base64 -w0 ~/.ssh/id_ed25519) ./run.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Load .env if present (silently, so it can be missing) ───────────────────
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

IMAGE="${CU_IMAGE:-ghcr.io/yggdrasil-technologies/cu-bench:latest}"
RESULTS_HOST="${CU_RESULTS_DIR:-${SCRIPT_DIR}/results}"
SSH_KEY_PATH="${SSH_KEY_PATH:-}"

mkdir -p "$RESULTS_HOST"

echo "================================================================"
echo "  CU Benchmark Suite"
echo "  Image:   $IMAGE"
echo "  Results: $RESULTS_HOST"
[ -n "${CU_UPLOAD_DEST:-}" ]    && echo "  Upload:  ${CU_UPLOAD_DEST}"
[ -n "${CU_UPLOAD_WEBHOOK:-}" ] && echo "  Webhook: ${CU_UPLOAD_WEBHOOK}"
echo "================================================================"
echo ""

# Build optional flag arrays (bash arrays avoid eval quoting issues)
EXTRA=()
[ -n "${CU_QUICK:-}"           ] && EXTRA+=(-e CU_QUICK=1)
[ -n "${CU_GPU_CLOCK:-}"       ] && EXTRA+=(-e "CU_GPU_CLOCK=${CU_GPU_CLOCK}")
[ -n "${CU_BENCH_MODEL:-}"     ] && EXTRA+=(-e "CU_BENCH_MODEL=${CU_BENCH_MODEL}")
[ -n "${CU_UPLOAD_DEST:-}"     ] && EXTRA+=(-e "CU_UPLOAD_DEST=${CU_UPLOAD_DEST}")
[ -n "${CU_UPLOAD_WEBHOOK:-}"  ] && EXTRA+=(-e "CU_UPLOAD_WEBHOOK=${CU_UPLOAD_WEBHOOK}")
[ -n "${CU_UPLOAD_KEY:-}"      ] && EXTRA+=(-e "CU_UPLOAD_KEY=${CU_UPLOAD_KEY}")

# Mount SSH key if SSH_KEY_PATH is set and file exists
if [ -n "$SSH_KEY_PATH" ] && [ -f "$SSH_KEY_PATH" ]; then
    EXTRA+=(-v "${SSH_KEY_PATH}:/root/.ssh/id_bench:ro")
    echo "  SSH key: ${SSH_KEY_PATH} → /root/.ssh/id_bench"
fi

exec docker run --rm \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --ipc=host \
    --ulimit memlock=-1:-1 \
    -v "${RESULTS_HOST}:/results" \
    "${EXTRA[@]}" \
    "$@" \
    "$IMAGE"
