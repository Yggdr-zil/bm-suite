#!/usr/bin/env python3
"""
CU Benchmark Suite — Memory Bandwidth Benchmark
Measures sustained HBM/GDDR bandwidth using tensor clone operations.
"""
import torch
import time
import os
import sys

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from _common import RESULTS_DIR, write_result, console, print_header, check_vram_ok

WARMUP_ITERS = int(os.environ.get("CU_MEMBW_WARMUP", "10"))
BENCH_ITERS = int(os.environ.get("CU_MEMBW_ITERS", "100"))


def bench_membw(size_gb, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    device = torch.device("cuda")
    numel = int(size_gb * 1e9 / 4)

    # clone needs src + dst = 2x tensor size
    if not check_vram_ok(numel * 4 * 2, label):
        return None

    try:
        src = torch.randn(numel, device=device, dtype=torch.float32)
    except RuntimeError as e:
        console.print(f"  [yellow]{label}: SKIP — cannot allocate ({e})[/]")
        return None

    with Progress(SpinnerColumn(), TextColumn(f"[cyan]{label}[/] warmup"), BarColumn(bar_width=20),
                   TextColumn("{task.completed}/{task.total}"), console=console, transient=True) as progress:
        task = progress.add_task("", total=warmup)
        for _ in range(warmup):
            dst = src.clone()
            torch.cuda.synchronize()
            progress.advance(task)

    times = []
    with Progress(SpinnerColumn(), TextColumn(f"[cyan]{label}[/] bench "), BarColumn(bar_width=30),
                   TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                   console=console, transient=True) as progress:
        task = progress.add_task("", total=iters)
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dst = src.clone()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            progress.advance(task)

    avg_time = sum(times) / len(times)
    bytes_moved = numel * 4 * 2
    gbps = bytes_moved / avg_time / 1e9

    del src, dst
    torch.cuda.empty_cache()

    return {
        "gbps": round(gbps, 1),
        "avg_ms": round(avg_time * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "tensor_gb": size_gb,
        "bytes_moved": bytes_moved,
    }


def bench_membw_mul(size_gb, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Element-wise multiply: read A, read B, write C = 3x tensor size."""
    device = torch.device("cuda")
    numel = int(size_gb * 1e9 / 4)

    # A + B + C = 3x tensor size
    if not check_vram_ok(numel * 4 * 3, label):
        return None

    try:
        A = torch.randn(numel, device=device, dtype=torch.float32)
        B = torch.randn(numel, device=device, dtype=torch.float32)
        C = torch.empty(numel, device=device, dtype=torch.float32)
    except RuntimeError as e:
        console.print(f"  [yellow]{label}: SKIP — cannot allocate ({e})[/]")
        return None

    with Progress(SpinnerColumn(), TextColumn(f"[cyan]{label}[/] bench "), BarColumn(bar_width=30),
                   TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                   console=console, transient=True) as progress:
        for _ in range(warmup):
            torch.mul(A, B, out=C)
        torch.cuda.synchronize()

        times = []
        task = progress.add_task("", total=iters)
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.mul(A, B, out=C)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            progress.advance(task)

    avg_time = sum(times) / len(times)
    bytes_moved = numel * 4 * 3
    gbps = bytes_moved / avg_time / 1e9

    del A, B, C
    torch.cuda.empty_cache()

    return {
        "gbps": round(gbps, 1),
        "avg_ms": round(avg_time * 1000, 3),
        "method": "element_mul",
        "tensor_gb": size_gb,
        "bytes_moved": bytes_moved,
    }


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device.[/]")
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print_header(f"Memory Bandwidth — {device_name}", f"{total_gb:.1f} GB VRAM")

    results = {"device": device_name, "total_vram_gb": round(total_gb, 1)}
    test_size = max(1.0, round(total_gb * 0.1))
    test_size_lg = max(2.0, round(total_gb * 0.25))

    results["clone_primary"] = bench_membw(test_size, f"Clone {test_size}GB")
    if results["clone_primary"]:
        console.print(f"  [bold green]Clone {test_size}GB:[/]  {results['clone_primary']['gbps']} GB/s")

    results["clone_large"] = bench_membw(test_size_lg, f"Clone {test_size_lg}GB")
    if results["clone_large"]:
        console.print(f"  [bold green]Clone {test_size_lg}GB:[/] {results['clone_large']['gbps']} GB/s")

    results["mul_primary"] = bench_membw_mul(test_size, f"Mul {test_size}GB")
    if results["mul_primary"]:
        console.print(f"  [bold green]Mul {test_size}GB:[/]    {results['mul_primary']['gbps']} GB/s")

    # Summary
    table = Table(title="\nMemory BW Results", show_lines=True, border_style="cyan")
    table.add_column("Test", style="bold")
    table.add_column("GB/s", justify="right", style="green")
    table.add_column("Avg ms", justify="right", style="dim")

    for key, label in [("clone_primary", f"Clone {test_size}GB"), ("clone_large", f"Clone {test_size_lg}GB"),
                        ("mul_primary", f"Mul {test_size}GB")]:
        r = results.get(key)
        if r:
            table.add_row(label, str(r["gbps"]), f"{r['avg_ms']:.3f}")
    console.print(table)

    write_result("02_membw.json", results)


if __name__ == "__main__":
    main()
