#!/bin/bash
# CU Benchmark Suite — Data Upload
# Sends results to your server before the container exits.
# The uploaded folder/tarball is named:
#   {provider}_{instance}_{silicon_id}
# The server-side receive.sh appends _{run_number} on arrival.
#
# Configure via environment variables (or .env loaded by run.sh):
#
#   CU_UPLOAD_DEST        rsync destination: user@host:/path/to/inbox/
#                         e.g. bearj@192.168.1.10:/data/cu-inbox/
#
#   CU_UPLOAD_KEY         (optional) base64-encoded SSH private key.
#                         If not set, uses mounted key at /root/.ssh/id_bench
#                         Mount with: -v ~/.ssh/id_ed25519:/root/.ssh/id_bench:ro
#
#   CU_UPLOAD_WEBHOOK     (optional) HTTPS URL — receives a POST with tarball.
#                         e.g. https://myserver.com/bench/upload
#
#   CU_PROVIDER           override auto-detected cloud provider name
#   CU_INSTANCE_NAME      override auto-detected instance/node name
#
# Upload failures are non-fatal: prints warning and exits 0.

set -uo pipefail

RESULTS_DIR="${RESULTS_DIR:-/results}"
KEY_FILE="/root/.ssh/id_bench"
DEST="${CU_UPLOAD_DEST:-}"
WEBHOOK="${CU_UPLOAD_WEBHOOK:-}"

if [ -z "$DEST" ] && [ -z "$WEBHOOK" ]; then
    echo "  [upload] No CU_UPLOAD_DEST or CU_UPLOAD_WEBHOOK set — skipping."
    exit 0
fi

# ─── Detect provider ─────────────────────────────────────────────────────────
detect_provider() {
    # Explicit override wins
    if [ -n "${CU_PROVIDER:-}" ]; then echo "$CU_PROVIDER"; return; fi

    # Lambda Labs: sets LAMBDA_TASK_ROOT or has lambda in hostname
    hostname 2>/dev/null | grep -qi "lambda" && echo "lambda" && return
    [ -f /etc/lambda_cloud ] && echo "lambda" && return

    # RunPod: RUNPOD_POD_ID always set
    [ -n "${RUNPOD_POD_ID:-}" ] && echo "runpod" && return

    # Vast.ai: VAST_CONTAINERLABEL set
    [ -n "${VAST_CONTAINERLABEL:-}" ] && echo "vast" && return

    # CoreWeave: COREWEAVE_CLOUD set or kubeconfig present
    [ -n "${COREWEAVE_CLOUD:-}" ] && echo "coreweave" && return

    # AWS: metadata service at 169.254.169.254
    if curl -sf --max-time 1 http://169.254.169.254/latest/meta-data/ >/dev/null 2>&1; then
        echo "aws"; return
    fi

    # GCP: metadata server at metadata.google.internal
    if curl -sf --max-time 1 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/id >/dev/null 2>&1; then
        echo "gcp"; return
    fi

    # Azure: IMDS endpoint
    if curl -sf --max-time 1 -H "Metadata: true" \
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01" >/dev/null 2>&1; then
        echo "azure"; return
    fi

    echo "unknown"
}

# ─── Detect instance name ─────────────────────────────────────────────────────
detect_instance() {
    if [ -n "${CU_INSTANCE_NAME:-}" ]; then echo "$CU_INSTANCE_NAME"; return; fi

    # RunPod: use pod ID (short)
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "pod${RUNPOD_POD_ID:0:8}"; return
    fi

    # Vast: use container label
    if [ -n "${VAST_CONTAINERLABEL:-}" ]; then
        echo "$VAST_CONTAINERLABEL" | tr '/' '-' | head -c 32; return
    fi

    # AWS: query instance type from IMDS
    AWS_TYPE=$(curl -sf --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)
    if [ -n "$AWS_TYPE" ]; then
        # e.g. p4d.24xlarge → p4d24xl
        echo "$AWS_TYPE" | tr '.' '-' | sed 's/xlarge/xl/g'; return
    fi

    # GCP: machine type
    GCP_TYPE=$(curl -sf --max-time 2 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | \
        rev | cut -d'/' -f1 | rev || true)
    if [ -n "$GCP_TYPE" ]; then echo "$GCP_TYPE"; return; fi

    # Fallback: hostname (sanitized)
    hostname 2>/dev/null | tr '.' '-' | head -c 32 || echo "node"
}

# ─── Detect silicon ID (GPU serial from environment JSON) ────────────────────
detect_silicon_id() {
    ENV_JSON="${RESULTS_DIR}/00_environment.json"
    if [ -f "$ENV_JSON" ]; then
        SERIAL=$(python3 -c "
import json, sys
try:
    d = json.load(open('${ENV_JSON}'))
    s = d.get('gpu_serials', 'unknown')
    # Take first serial if comma-separated list, strip whitespace
    print(s.split(',')[0].strip()[:16])
except: print('unknown')
" 2>/dev/null || echo "unknown")
        # Sanitize: keep alphanumeric and dashes
        echo "$SERIAL" | tr -cd 'A-Za-z0-9-' | head -c 16
    else
        # Fall back to GPU model slug from nvidia-smi
        nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | \
            head -1 | tr ' ' '-' | tr -cd 'A-Za-z0-9-' | head -c 20 || echo "unknown"
    fi
}

PROVIDER=$(detect_provider | tr -cd 'A-Za-z0-9-' | tr '[:upper:]' '[:lower:]')
INSTANCE=$(detect_instance | tr -cd 'A-Za-z0-9-' | tr '[:upper:]' '[:lower:]')
SILICON=$(detect_silicon_id)
UPLOAD_NAME="${PROVIDER}_${INSTANCE}_${SILICON}"

echo ""
echo "================================================================"
echo "  CU Benchmark — Uploading Results"
echo "  Name:    $UPLOAD_NAME"
echo "  Source:  $RESULTS_DIR"
echo "  (server will append _runN on receipt)"
echo "================================================================"

# ─── Prepare SSH key ─────────────────────────────────────────────────────────
SSH_KEY_LOADED=false
if [ -n "${CU_UPLOAD_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "$CU_UPLOAD_KEY" | base64 -d > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    SSH_KEY_LOADED=true
    echo "  SSH key: loaded from CU_UPLOAD_KEY"
elif [ -f "$KEY_FILE" ]; then
    chmod 600 "$KEY_FILE"
    SSH_KEY_LOADED=true
    echo "  SSH key: $KEY_FILE"
fi

# ─── rsync upload ─────────────────────────────────────────────────────────────
if [ -n "$DEST" ]; then
    if ! $SSH_KEY_LOADED; then
        echo "  [WARN] No SSH key — rsync skipped. Set CU_UPLOAD_KEY or mount to $KEY_FILE"
    else
        SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -i $KEY_FILE"
        REMOTE_HOST="${DEST%%:*}"
        REMOTE_HOST_ONLY="${REMOTE_HOST##*@}"

        echo "  Connecting to $REMOTE_HOST_ONLY..."
        if ssh $SSH_OPTS -o ConnectTimeout=10 "$REMOTE_HOST" "echo ok" >/dev/null 2>&1; then

            # Rsync into a subdirectory named by UPLOAD_NAME so server can find it
            REMOTE_BASE="${DEST%/}"
            REMOTE_PATH="${REMOTE_BASE}/${UPLOAD_NAME}/"
            echo "  Syncing → ${REMOTE_PATH}"
            if rsync -az --progress \
                -e "ssh $SSH_OPTS" \
                "${RESULTS_DIR}/" \
                "${REMOTE_PATH}"; then

                # Signal receiver: create a .ready marker so server knows transfer is complete
                ssh $SSH_OPTS "$REMOTE_HOST" "touch '${REMOTE_PATH}/.ready'" 2>/dev/null || true
                echo "  [OK] rsync complete → ${REMOTE_PATH}"
            else
                echo "  [WARN] rsync failed. Results remain at $RESULTS_DIR."
            fi
        else
            echo "  [WARN] Cannot reach $REMOTE_HOST_ONLY. Check hostname and SSH key."
        fi
    fi
fi

# ─── Webhook upload ──────────────────────────────────────────────────────────
if [ -n "$WEBHOOK" ]; then
    TARBALL_PATH="/tmp/${UPLOAD_NAME}.tar.gz"
    echo "  Packing → ${UPLOAD_NAME}.tar.gz..."
    if tar -czf "$TARBALL_PATH" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")" 2>/dev/null; then
        SIZE=$(du -sh "$TARBALL_PATH" | cut -f1)
        echo "  Packed: ${SIZE}"
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 300 \
            -X POST \
            -F "file=@${TARBALL_PATH};filename=${UPLOAD_NAME}.tar.gz" \
            -F "upload_name=${UPLOAD_NAME}" \
            -F "provider=${PROVIDER}" \
            -F "instance=${INSTANCE}" \
            -F "silicon_id=${SILICON}" \
            "${WEBHOOK}")
        if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
            echo "  [OK] Webhook upload complete (HTTP ${HTTP_CODE})"
        else
            echo "  [WARN] Webhook returned HTTP ${HTTP_CODE}."
        fi
        rm -f "$TARBALL_PATH"
    else
        echo "  [WARN] Could not create tarball."
    fi
fi

echo ""
