#!/usr/bin/env python3
"""
CU Benchmark Suite — Shared Runtime Configuration

Every benchmark script imports this to get:
  - RESULTS_DIR: auto-created directory for this run
  - RUN_ID: unique identifier (timestamp + short hash)
  - write_result(): saves JSON with run metadata baked in
  - console: rich Console instance

Usage in any benchmark script:
    from _common import RESULTS_DIR, RUN_ID, write_result, console, get_run_meta
"""
import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel

console = Console()

# ─── Run Identity ───────────────────────────────────────────────
_now = datetime.now(timezone.utc)
_TIMESTAMP = _now.strftime("%Y%m%dT%H%M%SZ")
_SHORT_HASH = hashlib.sha256(
    f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}".encode()
).hexdigest()[:8]

RUN_ID = f"{_TIMESTAMP}_{_SHORT_HASH}"

# ─── Results Directory ──────────────────────────────────────────
# Priority: $RESULTS_DIR env var > ./results/<run_id>
# If RESULTS_DIR is set AND already contains benchmark files, reuse it.
# Otherwise, create a timestamped subdirectory for isolation.
_explicit_dir = os.environ.get("RESULTS_DIR", "").strip()

if _explicit_dir:
    # User explicitly set it — use it directly, create if needed
    RESULTS_DIR = _explicit_dir
else:
    # Default: ./results/ relative to bm-suite root
    _bm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _results_root = os.path.join(_bm_root, "results")
    RESULTS_DIR = os.path.join(_results_root, f"run_{_TIMESTAMP}")

os.makedirs(RESULTS_DIR, exist_ok=True)
# Export so child processes (bash scripts) inherit it
os.environ["RESULTS_DIR"] = RESULTS_DIR


# ─── VRAM Budget ─────────────────────────────────────────────
# CU_VRAM_BUDGET: max fraction of FREE VRAM the suite is allowed to touch.
# Default 0.90 (90%) — leaves 10% headroom for system/compositor/other.
# Set to 1.0 for dedicated benchmark runs, or 0.5 if training alongside.
VRAM_BUDGET = float(os.environ.get("CU_VRAM_BUDGET", "0.90"))


def get_vram_budget_bytes(device_idx=0):
    """Return the max bytes the benchmark should allocate on this GPU.

    Uses free VRAM at call time × VRAM_BUDGET fraction.
    Returns (budget_bytes, free_bytes, total_bytes).
    """
    try:
        import torch
        free, total = torch.cuda.mem_get_info(device_idx)
    except Exception:
        return None, None, None
    budget = int(free * VRAM_BUDGET)
    return budget, free, total


def check_vram_ok(needed_bytes, label="", device_idx=0):
    """Check if we can safely allocate needed_bytes. Prints warning if tight.

    Returns True if allocation is within budget, False if it would exceed.
    """
    budget, free, total = get_vram_budget_bytes(device_idx)
    if budget is None:
        return True  # can't check, proceed anyway

    free_gb = free / (1024**3)
    need_gb = needed_bytes / (1024**3)
    budget_gb = budget / (1024**3)

    if needed_bytes > budget:
        console.print(
            f"  [bold yellow]SKIP {label}:[/] needs {need_gb:.1f} GB but budget is "
            f"{budget_gb:.1f} GB ({free_gb:.1f} GB free × {VRAM_BUDGET:.0%})"
        )
        console.print(
            f"  [dim]Set CU_VRAM_BUDGET=1.0 for full VRAM access, or free GPU memory[/]"
        )
        return False

    if needed_bytes > free * 0.8:
        console.print(
            f"  [yellow]WARNING {label}:[/] allocating {need_gb:.1f} GB "
            f"({free_gb:.1f} GB free) — tight on headroom"
        )
    return True


def get_run_meta():
    """Return metadata dict that gets embedded in every result file."""
    meta = {
        "run_id": RUN_ID,
        "timestamp": _now.isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "results_dir": RESULTS_DIR,
    }
    # Add PyTorch/CUDA info if available
    try:
        import torch
        meta["pytorch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda or "N/A"
        if torch.cuda.is_available():
            meta["gpu_model"] = torch.cuda.get_device_name(0)
            meta["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return meta


def write_result(filename, data):
    """Save benchmark result JSON with run metadata injected.

    Args:
        filename: e.g. "01_gemm.json"
        data: dict of benchmark-specific results

    The saved file will have:
        { "_meta": { run_id, timestamp, ... }, ...benchmark data... }
    """
    output = {"_meta": get_run_meta()}
    output.update(data)

    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"[dim]Saved: {path}[/]")
    return path


def print_header(title, subtitle=""):
    """Print a styled header panel for a benchmark section."""
    content = f"[bold]{title}[/]"
    if subtitle:
        content += f"\n{subtitle}"
    console.print()
    console.print(Panel(
        content,
        title=f"[bold cyan]CU Bench[/] [dim]|[/] [cyan]{RUN_ID[:15]}[/]",
        border_style="cyan",
        padding=(0, 2),
    ))


# ─── Startup Banner (only when run as main) ────────────────────
if os.environ.get("_CU_BENCH_QUIET") != "1":
    console.print(f"[dim]Run ID:[/] [bold]{RUN_ID}[/]  [dim]Results:[/] {RESULTS_DIR}")
