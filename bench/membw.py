#!/usr/bin/env python3
"""
CU Benchmark Suite — Memory Bandwidth Benchmark
Measures sustained HBM/GDDR bandwidth using tensor clone operations.
"""
import torch
import torch.multiprocessing as mp
import time
import json
import os
import sys

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from _common import RESULTS_DIR, write_result, console, print_header, check_vram_ok

WORKER_TIMEOUT = int(os.environ.get("CU_WORKER_TIMEOUT", "600"))
WARMUP_ITERS = int(os.environ.get("CU_MEMBW_WARMUP", "10"))
BENCH_ITERS = int(os.environ.get("CU_MEMBW_ITERS", "100"))


def bench_membw(size_gb, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_id=0):
    device = torch.device(f"cuda:{device_id}")
    numel = int(size_gb * 1e9 / 4)

    # clone needs src + dst = 2x tensor size
    if not check_vram_ok(numel * 4 * 2, label, device_idx=device_id):
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


def bench_membw_mul(size_gb, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_id=0):
    """Element-wise multiply: read A, read B, write C = 3x tensor size."""
    device = torch.device(f"cuda:{device_id}")
    numel = int(size_gb * 1e9 / 4)

    # A + B + C = 3x tensor size
    if not check_vram_ok(numel * 4 * 3, label, device_idx=device_id):
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


def _load_gpu_map():
    """Load gpu_map from 00_environment.json written by preflight.py.

    Returns (gpu_count, gpu_map) where gpu_map is a dict keyed by 'gpu0', 'gpu1', etc.
    Falls back to torch.cuda.device_count() with synthetic map if file is missing.
    """
    env_path = os.path.join(RESULTS_DIR, "00_environment.json")
    if os.path.exists(env_path):
        with open(env_path) as f:
            env_data = json.load(f)
        gpu_count = env_data.get("gpu_count", torch.cuda.device_count())
        gpu_map = env_data.get("gpu_map", {})
        if gpu_map:
            return gpu_count, gpu_map

    # Fallback: build synthetic map from torch
    gpu_count = torch.cuda.device_count()
    gpu_map = {}
    for i in range(gpu_count):
        gpu_map[f"gpu{i}"] = {
            "index": i,
            "uuid": f"unknown-{i}",
            "name": torch.cuda.get_device_name(i),
            "serial": "unknown",
        }
    return gpu_count, gpu_map


def _gpu_worker(gpu_id, barrier, result_dict, tests, warmup, iters):
    """Worker function for multi-GPU benchmarking.

    Each worker benchmarks all test types on its assigned GPU.
    """
    torch.cuda.set_device(gpu_id)
    gpu_results = {}

    for key, bench_fn, args in tests:
        try:
            if barrier is not None:
                barrier.wait(timeout=300)
            r = bench_fn(*args, warmup=warmup, iters=iters, device_id=gpu_id)
        except Exception:
            r = None
        gpu_results[key] = r

    result_dict[gpu_id] = gpu_results


def _run_multi_gpu(tests, warmup, iters, gpu_count, gpu_map):
    """Orchestrate membw benchmarks across all GPUs.

    Single GPU: runs directly, no multiprocessing overhead.
    Multi GPU: spawns one worker per GPU with barrier sync.

    Returns per_gpu dict (keyed by 'gpu0', etc.) with silicon_id and test results.
    """
    if gpu_count == 1:
        # Single GPU — no multiprocessing, run directly
        gpu_results = {}
        for key, bench_fn, args in tests:
            r = bench_fn(*args, warmup=warmup, iters=iters, device_id=0)
            gpu_results[key] = r
            if r:
                console.print(f"  [bold green]{key}:[/] {r['gbps']} GB/s  [dim]({r['avg_ms']:.2f} ms avg)[/]")

        # Build per_gpu with silicon_id
        gpu0_info = gpu_map.get("gpu0", {})
        per_gpu = {
            "gpu0": {
                "silicon_id": gpu0_info.get("uuid", "unknown"),
                **gpu_results,
            }
        }
        return per_gpu

    # Multi-GPU path: spawn workers with barrier sync
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()
    barrier = ctx.Barrier(gpu_count)

    processes = []
    for i in range(gpu_count):
        p = ctx.Process(
            target=_gpu_worker,
            args=(i, barrier, result_dict, tests, warmup, iters),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join(timeout=WORKER_TIMEOUT)

    failed = []
    for i, p in enumerate(processes):
        if p.exitcode is None:
            console.print(f"  [red]GPU{i} worker timed out — terminating[/]")
            p.terminate()
            p.join(timeout=10)
            failed.append(i)
        elif p.exitcode != 0:
            console.print(f"  [red]GPU{i} worker crashed (exit={p.exitcode})[/]")
            failed.append(i)

    # Build per_gpu dict with silicon_ids
    per_gpu = {}
    for i in range(gpu_count):
        gpu_key = f"gpu{i}"
        gpu_info = gpu_map.get(gpu_key, {})
        gpu_results = dict(result_dict.get(i, {}))
        per_gpu[gpu_key] = {
            "silicon_id": gpu_info.get("uuid", "unknown"),
            **gpu_results,
        }

    return per_gpu


def _aggregate_cluster(per_gpu, test_keys):
    """Build top-level test results with cluster_gbps from per_gpu data.

    For each test, picks gpu0's result as the representative (for top-level
    avg_ms, etc.) and sums gbps across all GPUs for cluster_gbps.

    Returns dict keyed by test name with cluster_gbps added.
    """
    aggregated = {}
    for key in test_keys:
        # Collect all non-None results for this test
        gpu_results = []
        for gpu_key in sorted(per_gpu.keys()):
            gpu_data = per_gpu[gpu_key]
            r = gpu_data.get(key)
            if r is not None:
                gpu_results.append(r)

        if not gpu_results:
            aggregated[key] = None
            continue

        # Use gpu0's result as representative, add cluster_gbps
        representative = dict(gpu_results[0])
        cluster_gbps = round(sum(r["gbps"] for r in gpu_results), 1)
        representative["cluster_gbps"] = cluster_gbps

        # For multi-GPU, gbps is the average per-GPU gbps
        if len(gpu_results) > 1:
            representative["gbps"] = round(
                sum(r["gbps"] for r in gpu_results) / len(gpu_results), 1
            )

        aggregated[key] = representative
    return aggregated


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device.[/]")
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_count, gpu_map = _load_gpu_map()

    print_header(
        f"Memory Bandwidth — {device_name}",
        f"{total_gb:.1f} GB VRAM | Warmup: {WARMUP_ITERS} | Iters: {BENCH_ITERS} | GPUs: {gpu_count}",
    )

    test_size = max(1.0, round(total_gb * 0.1))
    test_size_lg = max(2.0, round(total_gb * 0.25))

    tests = [
        ("clone_primary", bench_membw, (test_size, f"Clone {test_size}GB")),
        ("clone_large", bench_membw, (test_size_lg, f"Clone {test_size_lg}GB")),
        ("mul_primary", bench_membw_mul, (test_size, f"Mul {test_size}GB")),
    ]

    per_gpu = _run_multi_gpu(tests, WARMUP_ITERS, BENCH_ITERS, gpu_count, gpu_map)

    # Aggregate cluster-level results
    test_keys = [key for key, _, _ in tests]
    aggregated = _aggregate_cluster(per_gpu, test_keys)

    # Build output
    results = {
        "device": device_name,
        "total_vram_gb": round(total_gb, 1),
        "gpu_count": gpu_count,
    }
    results.update(aggregated)
    results["per_gpu"] = per_gpu

    # Multi-GPU: print summary (single-GPU already printed inline)
    if gpu_count > 1:
        for key, _, _ in tests:
            r = aggregated.get(key)
            if r:
                console.print(
                    f"  [bold green]{key}:[/] {r['cluster_gbps']} GB/s (cluster)  "
                    f"[dim]({r['gbps']} avg/GPU)[/]"
                )

    # Summary table
    table = Table(title="\nMemory BW Results", show_lines=True, border_style="cyan")
    table.add_column("Test", style="bold")
    table.add_column("GB/s", justify="right", style="green")
    if gpu_count > 1:
        table.add_column("Cluster", justify="right", style="bold green")
    table.add_column("Avg ms", justify="right", style="dim")

    for key, _, _ in tests:
        r = results.get(key)
        if r:
            row = [key]
            row.append(str(r["gbps"]))
            if gpu_count > 1:
                row.append(str(r["cluster_gbps"]))
            row.append(f"{r['avg_ms']:.3f}")
            table.add_row(*row)
        else:
            empty_count = 3 if gpu_count == 1 else 4
            empty = [""] * empty_count
            empty[0] = key
            empty[1] = "[dim]skipped[/]"
            table.add_row(*empty)

    console.print(table)
    write_result("02_membw.json", results)


if __name__ == "__main__":
    main()
