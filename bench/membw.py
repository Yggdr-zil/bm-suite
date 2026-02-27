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
BENCH_ITERS = int(os.environ.get("CU_MEMBW_ITERS", "5000" if not os.environ.get("CU_QUICK") else "500"))
TRIM_PCT = float(os.environ.get("CU_MEMBW_TRIM_PCT", "5"))


def bench_membw(size_gb, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_idx=0):
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)
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

    sorted_times = sorted(times)
    trim_count = max(1, int(len(times) * TRIM_PCT / 100))
    trimmed = sorted_times[:-trim_count]

    avg_time = sum(trimmed) / len(trimmed)
    bytes_moved = numel * 4 * 2
    gbps = bytes_moved / avg_time / 1e9

    n = len(sorted_times)
    def pct(p): return round(sorted_times[int(n * p / 100)] * 1000, 3)
    timing_percentiles_ms = {
        "p1": pct(1), "p5": pct(5), "p25": pct(25),
        "p50": pct(50), "p75": pct(75), "p95": pct(95), "p99": pct(99),
    }
    sample_stride = max(1, iters // 100)
    sampled_gbps = [
        round(bytes_moved / times[i] / 1e9, 1)
        for i in range(0, len(times), sample_stride)
    ]

    del src, dst
    torch.cuda.empty_cache()

    return {
        "gbps": round(gbps, 1),
        "avg_ms": round(avg_time * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "tensor_gb": size_gb,
        "bytes_moved": bytes_moved,
        "total_samples": len(times),
        "trim_pct": TRIM_PCT,
        "timing_percentiles_ms": timing_percentiles_ms,
        "sampled_gbps": sampled_gbps,
    }


def bench_membw_mul(size_gb, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_idx=0):
    """Element-wise multiply: read A, read B, write C = 3x tensor size."""
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)
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

    sorted_times = sorted(times)
    trim_count = max(1, int(len(times) * TRIM_PCT / 100))
    trimmed = sorted_times[:-trim_count]

    avg_time = sum(trimmed) / len(trimmed)
    bytes_moved = numel * 4 * 3
    gbps = bytes_moved / avg_time / 1e9

    n = len(sorted_times)
    def pct(p): return round(sorted_times[int(n * p / 100)] * 1000, 3)

    del A, B, C
    torch.cuda.empty_cache()

    return {
        "gbps": round(gbps, 1),
        "avg_ms": round(avg_time * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "method": "element_mul",
        "tensor_gb": size_gb,
        "bytes_moved": bytes_moved,
        "total_samples": len(times),
        "trim_pct": TRIM_PCT,
        "timing_percentiles_ms": {
            "p1": pct(1), "p5": pct(5), "p25": pct(25),
            "p50": pct(50), "p75": pct(75), "p95": pct(95), "p99": pct(99),
        },
    }


def _bench_one_gpu_membw(device_idx):
    """Run all membw tests on a single GPU. Returns dict of test results."""
    total_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
    test_size    = max(1.0, round(total_gb * 0.1))
    test_size_lg = max(2.0, round(total_gb * 0.25))

    r_primary = bench_membw(test_size,    f"GPU{device_idx}/Clone{test_size}GB",    device_idx=device_idx)
    r_large   = bench_membw(test_size_lg, f"GPU{device_idx}/Clone{test_size_lg}GB", device_idx=device_idx)
    r_mul     = bench_membw_mul(test_size, f"GPU{device_idx}/Mul{test_size}GB",     device_idx=device_idx)

    for label, r in [(f"Clone {test_size}GB", r_primary), (f"Clone {test_size_lg}GB", r_large),
                      (f"Mul {test_size}GB", r_mul)]:
        if r:
            console.print(f"  [bold]GPU {device_idx}[/] [green]{label}:[/] {r['gbps']} GB/s")

    return {
        "total_vram_gb": round(total_gb, 1),
        "clone_primary": r_primary,
        "clone_large":   r_large,
        "mul_primary":   r_mul,
    }


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device.[/]")
        sys.exit(0)

    gpu_count   = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    total_gb    = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print_header(f"Memory Bandwidth — {device_name} × {gpu_count}",
                 f"{total_gb:.1f} GB VRAM per GPU | {gpu_count} GPU(s)")

    results = {
        "device": device_name,
        "gpu_count": gpu_count,
        "gpus": {},
    }

    for dev_idx in range(gpu_count):
        dev_name = torch.cuda.get_device_name(dev_idx)
        console.print(f"\n[bold cyan]── GPU {dev_idx}: {dev_name} ──[/]")
        gpu_res = _bench_one_gpu_membw(dev_idx)
        results["gpus"][f"gpu_{dev_idx}"] = {"device_name": dev_name, **gpu_res}

    # GPU 0 floated to top level for backward compat with report.py
    gpu0 = results["gpus"].get("gpu_0", {})
    results["clone_primary"] = gpu0.get("clone_primary")
    results["clone_large"]   = gpu0.get("clone_large")
    results["mul_primary"]   = gpu0.get("mul_primary")
    results["total_vram_gb"] = gpu0.get("total_vram_gb")

    # Summary table — one row per GPU
    test_size    = max(1.0, round(total_gb * 0.1))
    test_size_lg = max(2.0, round(total_gb * 0.25))
    table = Table(title="\nMemory BW Results (all GPUs)", show_lines=True, border_style="cyan")
    table.add_column("GPU", style="bold")
    table.add_column(f"Clone {test_size}GB", justify="right", style="green")
    table.add_column(f"Clone {test_size_lg}GB", justify="right", style="green")
    table.add_column(f"Mul {test_size}GB", justify="right", style="dim")
    table.add_column("p50 ms", justify="right", style="dim")
    table.add_column("p99 ms", justify="right", style="dim")

    for dev_idx in range(gpu_count):
        gr = results["gpus"].get(f"gpu_{dev_idx}", {})
        def gbps(k): return str(gr[k]["gbps"]) if gr.get(k) else "-"
        pcts = (gr.get("clone_large") or {}).get("timing_percentiles_ms", {})
        table.add_row(
            f"GPU {dev_idx}", gbps("clone_primary"), gbps("clone_large"),
            gbps("mul_primary"), str(pcts.get("p50", "-")), str(pcts.get("p99", "-"))
        )
    console.print(table)

    write_result("02_membw.json", results)


if __name__ == "__main__":
    main()
