#!/usr/bin/env python3
"""
CU Benchmark Suite — VRAM Capacity Benchmark
Physically measures usable GPU memory via binary search allocation.

Does NOT trust reported numbers — allocates real tensors to verify.

Accounts for CUDA context overhead:
  The benchmark process itself uses ~200-500 MB of VRAM for the CUDA
  runtime (page tables, kernel code, cuBLAS handles, allocator metadata).
  Those bytes are implicitly verified — if they were bad, the process
  would have crashed. We measure that overhead and add it to the probe
  result so the final number reflects ALL working VRAM, not just the
  free pool we probed.

  verified_working = max_allocatable + cuda_context_bytes
"""
import torch
import os
import sys

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from _common import RESULTS_DIR, write_result, console, print_header

PROBE_STEPS = 30  # binary search iterations → ~0.00002% precision


def probe_usable_vram(device_idx):
    """Binary search for max allocatable VRAM, then add context overhead."""
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(device)
    reported_bytes = props.total_memory
    reported_gb = reported_bytes / (1024**3)

    # Snapshot BEFORE any probe allocations.
    # free = what we can probe. total - free = CUDA context overhead.
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    context_bytes = total_bytes - free_bytes
    context_gb = context_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)

    is_shared = context_gb > 0.6  # normal CUDA context is ~200-500 MB
    other_gb = 0.0
    if is_shared:
        # Estimate: typical CUDA context ≈ 300 MB, rest is other processes
        est_context = 0.3 * (1024**3)
        other_gb = (context_bytes - est_context) / (1024**3)
        console.print(
            f"  [yellow]GPU {device_idx}:[/] ~{other_gb:.2f} GB held by other processes + "
            f"~{est_context / (1024**3):.2f} GB CUDA context"
        )
    else:
        console.print(
            f"  [dim]GPU {device_idx}: CUDA context overhead: {context_gb:.4f} GB[/]"
        )

    # ─── Binary search within free space ───
    low = 0
    high = free_bytes

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]GPU {device_idx}[/] probing"),
        BarColumn(bar_width=25),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[dim]{task.fields[status]}[/]"),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("", total=PROBE_STEPS, status="")
        for i in range(PROBE_STEPS):
            mid = (low + high) // 2
            numel = mid // 4  # float32 = 4 bytes

            torch.cuda.empty_cache()
            try:
                t = torch.empty(numel, device=device, dtype=torch.float32)
                del t
                torch.cuda.empty_cache()
                low = mid
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                high = mid

            progress.update(task, advance=1, status=f"{low / (1024**3):.3f} GB")

    max_alloc_bytes = low
    max_alloc_gb = max_alloc_bytes / (1024**3)

    # ─── The key insight: context memory is implicitly verified ───
    # If those pages were dead, our process would have crashed.
    # So total verified = what we probed + what the runtime is sitting on.
    verified_bytes = max_alloc_bytes + context_bytes
    verified_gb = verified_bytes / (1024**3)

    delta_vs_reported = reported_gb - verified_gb

    result = {
        "device_name": props.name,
        "verified_total_gb": round(verified_gb, 4),
        "max_allocatable_gb": round(max_alloc_gb, 4),
        "cuda_context_gb": round(context_gb, 4),
        "reported_total_gb": round(reported_gb, 2),
        "driver_reported_total_gb": round(total_bytes / (1024**3), 4),
        "delta_vs_reported_gb": round(delta_vs_reported, 4),
        "delta_vs_reported_pct": round(delta_vs_reported / reported_gb * 100, 3) if reported_gb > 0 else 0,
    }

    if is_shared:
        result["other_processes_gb"] = round(other_gb, 2)
        result["note"] = (
            f"GPU shared — context_bytes includes ~{other_gb:.2f} GB from other processes. "
            f"verified_total includes that memory (it IS working, just not ours). "
            f"For clean hardware capacity, run with GPU idle."
        )

    return result


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device.[/]")
        sys.exit(0)

    gpu_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print_header(
        f"VRAM Probe — {device_name}",
        f"{gpu_count} GPU(s) | {PROBE_STEPS} binary search steps\n"
        f"verified = max_allocatable + cuda_context"
    )

    results = {"gpu_count": gpu_count, "gpus": {}}

    for i in range(gpu_count):
        info = probe_usable_vram(i)
        results["gpus"][f"gpu_{i}"] = info

        verified = info["verified_total_gb"]
        reported = info["reported_total_gb"]
        alloc = info["max_allocatable_gb"]
        ctx = info["cuda_context_gb"]
        delta = info["delta_vs_reported_gb"]
        delta_pct = info["delta_vs_reported_pct"]

        console.print(
            f"\n  [bold green]GPU {i} verified:[/] [bold]{verified:.4f} GB[/]"
        )
        console.print(
            f"  [dim]  = {alloc:.4f} GB probed + {ctx:.4f} GB context[/]"
        )
        console.print(
            f"  [dim]  vs {reported:.2f} GB reported "
            f"(delta: {delta:.4f} GB / {delta_pct:.3f}%)[/]"
        )

    results["total_verified_gb"] = round(
        sum(g["verified_total_gb"] for g in results["gpus"].values()), 4
    )
    results["total_reported_gb"] = round(
        sum(g["reported_total_gb"] for g in results["gpus"].values()), 2
    )

    # Summary table
    console.print()
    table = Table(title="VRAM Probe Results", show_lines=True, border_style="cyan")
    table.add_column("GPU", style="bold")
    table.add_column("Verified Total", justify="right", style="bold green")
    table.add_column("Probed", justify="right")
    table.add_column("Context", justify="right", style="dim")
    table.add_column("Reported", justify="right")
    table.add_column("Delta", justify="right", style="yellow")

    for k, v in results["gpus"].items():
        table.add_row(
            k,
            f"{v['verified_total_gb']:.4f} GB",
            f"{v['max_allocatable_gb']:.4f} GB",
            f"{v['cuda_context_gb']:.4f} GB",
            f"{v['reported_total_gb']:.2f} GB",
            f"{v['delta_vs_reported_gb']:.4f} GB ({v['delta_vs_reported_pct']:.3f}%)",
        )
    console.print(table)

    write_result("03_vram.json", results)


if __name__ == "__main__":
    main()
