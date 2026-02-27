#!/bin/bash
# CU Benchmark Suite — Server-Side Receiver
# Watches an inbox directory for completed uploads and renames them:
#   {provider}_{instance}_{silicon_id}  →  {provider}_{instance}_{silicon_id}_{runN}
#
# The sender (upload.sh) creates a .ready marker when rsync is done.
# This script watches for .ready files, assigns the next run number,
# renames the directory, and moves it to the archive.
#
# ─── Setup ───────────────────────────────────────────────────────────────────
#   mkdir -p ~/cu-bench/{inbox,archive,logs}
#   cp receive.sh ~/cu-bench/
#   chmod +x ~/cu-bench/receive.sh
#
# ─── Run as a service ─────────────────────────────────────────────────────────
#   # Option A: run in a screen/tmux session
#   screen -S cu-receive bash ~/cu-bench/receive.sh
#
#   # Option B: systemd service (recommended for always-on home server)
#   # See: server/cu-receive.service
#
# ─── Directory layout ────────────────────────────────────────────────────────
#   INBOX/                    ← upload.sh rsyncs here
#     aws_p4d24xl_GPU0001/    ← named by sender
#       benchmark_report.json
#       plots/
#       .ready                ← created by upload.sh when transfer complete
#   ARCHIVE/                  ← receiver moves completed runs here
#     aws_p4d24xl_GPU0001_run1/
#     aws_p4d24xl_GPU0001_run2/
#     lambda_a100-80g_GPU0042_run1/
#   COUNTER_FILE              ← JSON: {"aws_p4d24xl_GPU0001": 3, ...}
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
INBOX="${CU_INBOX:-${BASE_DIR}/inbox}"
ARCHIVE="${CU_ARCHIVE:-${BASE_DIR}/archive}"
LOGS="${CU_LOGS:-${BASE_DIR}/logs}"
COUNTER_FILE="${BASE_DIR}/run_counters.json"
POLL_INTERVAL=5   # seconds between inbox scans

mkdir -p "$INBOX" "$ARCHIVE" "$LOGS"

# ─── Counter helpers ──────────────────────────────────────────────────────────
counter_get() {
    # Get current counter for a name, default 0
    local name="$1"
    if [ -f "$COUNTER_FILE" ]; then
        python3 -c "
import json
d = json.load(open('${COUNTER_FILE}'))
print(d.get('${name}', 0))
" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

counter_increment() {
    # Atomically increment counter for name, return new value
    local name="$1"
    local lockfile="${COUNTER_FILE}.lock"

    # Acquire lock (up to 10 seconds)
    local attempts=0
    while ! mkdir "$lockfile" 2>/dev/null; do
        sleep 0.2
        attempts=$((attempts + 1))
        if [ $attempts -gt 50 ]; then
            echo "ERROR: could not acquire counter lock" >&2
            return 1
        fi
    done

    # Read → increment → write
    local current new_val
    current=$(counter_get "$name")
    new_val=$((current + 1))

    python3 -c "
import json, os
path = '${COUNTER_FILE}'
d = json.load(open(path)) if os.path.exists(path) else {}
d['${name}'] = ${new_val}
json.dump(d, open(path, 'w'), indent=2, sort_keys=True)
print(${new_val})
" 2>/dev/null || echo "$new_val"

    rm -rf "$lockfile"
}

# ─── Process a single completed upload ───────────────────────────────────────
process_upload() {
    local upload_dir="$1"
    local base_name
    base_name=$(basename "$upload_dir")
    local ready_file="${upload_dir}/.ready"

    # Double-check .ready still exists (another process could have grabbed it)
    [ -f "$ready_file" ] || return 0

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Processing: ${base_name}" | tee -a "${LOGS}/receive.log"

    # Assign next run number
    local run_num
    run_num=$(counter_increment "$base_name")
    local final_name="${base_name}_run${run_num}"
    local dest="${ARCHIVE}/${final_name}"

    # Remove .ready before moving (so we don't double-process on error)
    rm -f "$ready_file"

    # Move to archive with final name
    if mv "$upload_dir" "$dest"; then
        echo "  → ${final_name}" | tee -a "${LOGS}/receive.log"

        # Write a receipt file into the archived run
        cat > "${dest}/.receipt.json" <<EOF
{
  "original_name": "${base_name}",
  "final_name": "${final_name}",
  "run_number": ${run_num},
  "received_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "archived_path": "${dest}"
}
EOF
        echo "  [OK] Archived: ${dest}"
    else
        echo "  [ERROR] Failed to move ${upload_dir} → ${dest}" | tee -a "${LOGS}/receive.log"
        # Put .ready back so it will be retried
        touch "$ready_file"
    fi
}

# ─── Main loop ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  CU Bench Receiver"
echo "  Inbox:   $INBOX"
echo "  Archive: $ARCHIVE"
echo "  Poll:    every ${POLL_INTERVAL}s"
echo "  $(date -u)"
echo "================================================================"

while true; do
    # Find all .ready markers in inbox
    while IFS= read -r -d '' ready_file; do
        upload_dir=$(dirname "$ready_file")
        process_upload "$upload_dir"
    done < <(find "$INBOX" -maxdepth 2 -name ".ready" -print0 2>/dev/null)

    sleep "$POLL_INTERVAL"
done
