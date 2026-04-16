#!/usr/bin/env python3
"""
CU Benchmark Suite — GEMM Benchmark
Measures sustained floating-point throughput at FP32, FP16, BF16, FP8.

Method: Square matrix multiply (M=N=K=8192) saturates tensor cores.
TFLOPS = 2 * M * N * K / time / 1e12
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

MATRIX_DIM = int(os.environ.get("CU_GEMM_DIM", "8192"))
WARMUP_ITERS = int(os.environ.get("CU_GEMM_WARMUP", "20"))
BENCH_ITERS = int(os.environ.get("CU_GEMM_ITERS", "200"))
# Drop the slowest N% of samples (cuBLAS auto-tuner tail)
TRIM_PCT = float(os.environ.get("CU_GEMM_TRIM_PCT", "5"))


def bench_gemm(M, N, K, dtype, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_id=0):
    """Run GEMM benchmark for a given precision."""
    device = torch.device(f"cuda:{device_id}")

    # Check VRAM budget before allocating (2 matrices + 1 output)
    elem_bytes = torch.finfo(dtype).bits // 8
    needed = (M * K + K * N + M * N) * elem_bytes
    if not check_vram_ok(needed, label, device_idx=device_id):
        return None

    try:
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
    except RuntimeError as e:
        console.print(f"  [yellow]{label}: SKIP — cannot allocate ({e})[/]")
        return None

    # Warmup: primes cuBLAS kernel auto-tuner + instruction cache
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{label}[/] warmup"),
        BarColumn(bar_width=20),
        TextColumn("{task.completed}/{task.total}"),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("warmup", total=warmup)
        for _ in range(warmup):
            C = torch.mm(A, B)
            torch.cuda.synchronize()
            progress.advance(task)

    # Measured iterations
    times = []
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{label}[/] bench "),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("bench", total=iters)
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            C = torch.mm(A, B)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            progress.advance(task)

    # Trimmed mean: drop the slowest TRIM_PCT% of samples
    # These are typically cuBLAS auto-tuner stragglers or thermal transients
    sorted_times = sorted(times)
    trim_count = max(1, int(len(times) * TRIM_PCT / 100))
    trimmed = sorted_times[:-trim_count]  # drop the N slowest

    avg_time = sum(trimmed) / len(trimmed)
    raw_avg = sum(times) / len(times)
    median_time = sorted_times[len(sorted_times) // 2]
    ops = 2 * M * N * K
    tflops = ops / avg_time / 1e12
    median_tflops = ops / median_time / 1e12
    std_time = (sum((t - avg_time)**2 for t in trimmed) / len(trimmed))**0.5
    cv_pct = round(std_time / avg_time * 100, 2)

    # Cleanup
    del A, B, C
    torch.cuda.empty_cache()

    # Use median when CV is high (power-throttling causes bimodal distribution)
    primary_tflops = median_tflops if cv_pct > 5.0 else tflops

    result = {
        "tflops": round(primary_tflops, 2),
        "avg_ms": round(avg_time * 1000, 3),
        "median_ms": round(median_time * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "std_ms": round(std_time * 1000, 3),
        "cv_pct": cv_pct,
        "raw_avg_ms": round(raw_avg * 1000, 3),
        "trimmed_samples": len(trimmed),
        "total_samples": len(times),
        "trim_pct": TRIM_PCT,
        "matrix_dim": M,
        "warmup_iters": warmup,
        "bench_iters": iters,
    }
    if cv_pct > 5.0:
        result["tflops_method"] = "median"
        result["tflops_trimmed_mean"] = round(tflops, 2)
    return result


def bench_gemm_fp8(M, N, K, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_id=0):
    """FP8 GEMM uses torch._scaled_mm (Hopper+/Blackwell only)."""
    device = torch.device(f"cuda:{device_id}")

    # FP8 needs intermediate FP16 + FP8 copies — budget ~4 bytes/element
    needed = (M * K + K * N + M * N) * 4
    if not check_vram_ok(needed, "FP8", device_idx=device_id):
        return None

    try:
        # cuBLASLt requires: A = row-major (M,K), B = column-major (K,N)
        # Column-major (K,N) = transposed view of contiguous (N,K)
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn).t()
        scale_a = torch.ones(1, device=device, dtype=torch.float32)
        scale_b = torch.ones(1, device=device, dtype=torch.float32)
    except (RuntimeError, AttributeError) as e:
        console.print(f"  [yellow]FP8: SKIP — not supported ({e})[/]")
        return None

    try:
        # Warmup
        with Progress(SpinnerColumn(), TextColumn("[cyan]FP8[/] warmup"), BarColumn(bar_width=20),
                       TextColumn("{task.completed}/{task.total}"), console=console, transient=True) as progress:
            task = progress.add_task("warmup", total=warmup)
            for _ in range(warmup):
                C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
                torch.cuda.synchronize()
                progress.advance(task)

        # Measured
        times = []
        with Progress(SpinnerColumn(), TextColumn("[cyan]FP8[/] bench "), BarColumn(bar_width=30),
                       TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                       console=console, transient=True) as progress:
            task = progress.add_task("bench", total=iters)
            for _ in range(iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
                progress.advance(task)

        # Trimmed mean (same as standard GEMM)
        sorted_times = sorted(times)
        trim_count = max(1, int(len(times) * TRIM_PCT / 100))
        trimmed = sorted_times[:-trim_count]

        avg_time = sum(trimmed) / len(trimmed)
        raw_avg = sum(times) / len(times)
        median_time = sorted_times[len(sorted_times) // 2]
        ops = 2 * M * N * K
        tflops = ops / avg_time / 1e12
        median_tflops = ops / median_time / 1e12
        std_time = (sum((t - avg_time)**2 for t in trimmed) / len(trimmed))**0.5
        cv_pct = round(std_time / avg_time * 100, 2)

        del A, B, C
        torch.cuda.empty_cache()

        primary_tflops = median_tflops if cv_pct > 5.0 else tflops

        result = {
            "tflops": round(primary_tflops, 2),
            "avg_ms": round(avg_time * 1000, 3),
            "median_ms": round(median_time * 1000, 3),
            "min_ms": round(min(times) * 1000, 3),
            "max_ms": round(max(times) * 1000, 3),
            "std_ms": round(std_time * 1000, 3),
            "cv_pct": cv_pct,
            "raw_avg_ms": round(raw_avg * 1000, 3),
            "trimmed_samples": len(trimmed),
            "total_samples": len(times),
            "trim_pct": TRIM_PCT,
            "matrix_dim": M,
            "warmup_iters": warmup,
            "bench_iters": iters,
            "note": "scaled_mm (FP8 E4M3FN, scale=1.0)"
        }
        if cv_pct > 5.0:
            result["tflops_method"] = "median"
            result["tflops_trimmed_mean"] = round(tflops, 2)
        return result
    except Exception as e:
        console.print(f"  [yellow]FP8: SKIP — scaled_mm failed ({e})[/]")
        try:
            del A, B
        except NameError:
            pass
        torch.cuda.empty_cache()
        return None


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


def _gpu_worker(gpu_id, barrier, result_dict, precisions, M, N, K, warmup, iters):
    """Worker function for multi-GPU benchmarking.

    Each worker benchmarks all precisions on its assigned GPU.
    """
    torch.cuda.set_device(gpu_id)
    gpu_results = {}

    for key, dtype, label in precisions:
        if barrier is not None:
            barrier.wait()
        r = bench_gemm(M, N, K, dtype, f"GPU{gpu_id} {label}",
                       warmup=warmup, iters=iters, device_id=gpu_id)
        gpu_results[key] = r

    # FP8
    if barrier is not None:
        barrier.wait()
    r = bench_gemm_fp8(M, N, K, warmup=warmup, iters=iters, device_id=gpu_id)
    gpu_results["fp8"] = r

    result_dict[gpu_id] = gpu_results


def _run_multi_gpu(precisions, M, N, K, warmup, iters, gpu_count, gpu_map):
    """Orchestrate GEMM benchmarks across all GPUs.

    Single GPU: runs directly, no multiprocessing overhead.
    Multi GPU: spawns one worker per GPU with barrier sync.

    Returns per_gpu dict (keyed by 'gpu0', etc.) with silicon_id and precision results.
    """
    if gpu_count == 1:
        # Single GPU — no multiprocessing, run directly
        gpu_results = {}
        for key, dtype, label in precisions:
            r = bench_gemm(M, N, K, dtype, label, warmup=warmup, iters=iters, device_id=0)
            gpu_results[key] = r
            if r:
                method = f" [yellow](median, CV={r['cv_pct']}%)[/]" if r.get("tflops_method") == "median" else ""
                console.print(f"  [bold green]{label}:[/] {r['tflops']} TFLOPS  [dim]({r['avg_ms']:.2f} ms avg)[/]{method}")

        r = bench_gemm_fp8(M, N, K, warmup=warmup, iters=iters, device_id=0)
        gpu_results["fp8"] = r
        if r:
            method = f" [yellow](median, CV={r['cv_pct']}%)[/]" if r.get("tflops_method") == "median" else ""
            console.print(f"  [bold green]FP8:[/]  {r['tflops']} TFLOPS  [dim]({r['avg_ms']:.2f} ms avg)[/]{method}")

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
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = mp.Barrier(gpu_count)

    processes = []
    for i in range(gpu_count):
        p = mp.Process(
            target=_gpu_worker,
            args=(i, barrier, result_dict, precisions, M, N, K, warmup, iters),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

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


def _aggregate_cluster(per_gpu, precision_keys):
    """Build top-level precision results with cluster_tflops from per_gpu data.

    For each precision, picks gpu0's result as the representative (for top-level
    avg_ms, median_ms, etc.) and sums tflops across all GPUs for cluster_tflops.

    Returns dict keyed by precision with cluster_tflops added.
    """
    aggregated = {}
    for key in precision_keys:
        # Collect all non-None results for this precision
        gpu_results = []
        for gpu_key in sorted(per_gpu.keys()):
            gpu_data = per_gpu[gpu_key]
            r = gpu_data.get(key)
            if r is not None:
                gpu_results.append(r)

        if not gpu_results:
            aggregated[key] = None
            continue

        # Use gpu0's result as representative, add cluster_tflops
        representative = dict(gpu_results[0])
        cluster_tflops = round(sum(r["tflops"] for r in gpu_results), 2)
        representative["cluster_tflops"] = cluster_tflops

        # For multi-GPU, tflops is the average per-GPU tflops
        if len(gpu_results) > 1:
            representative["tflops"] = round(
                sum(r["tflops"] for r in gpu_results) / len(gpu_results), 2
            )

        aggregated[key] = representative
    return aggregated


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device. Skipping GEMM benchmark.[/]")
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    gpu_count, gpu_map = _load_gpu_map()

    print_header(
        f"{device_name}",
        f"Matrix: {MATRIX_DIM}x{MATRIX_DIM} | Warmup: {WARMUP_ITERS} | Iters: {BENCH_ITERS} | GPUs: {gpu_count}",
    )

    M = N = K = MATRIX_DIM

    precisions = [
        ("fp32", torch.float32, "FP32"),
        ("fp16", torch.float16, "FP16"),
        ("bf16", torch.bfloat16, "BF16"),
    ]

    per_gpu = _run_multi_gpu(precisions, M, N, K, WARMUP_ITERS, BENCH_ITERS, gpu_count, gpu_map)

    # Aggregate cluster-level results
    precision_keys = [key for key, _, _ in precisions] + ["fp8"]
    aggregated = _aggregate_cluster(per_gpu, precision_keys)

    # Build output
    results = {
        "device": device_name,
        "matrix_dim": M,
        "gpu_count": gpu_count,
    }
    results.update(aggregated)
    results["per_gpu"] = per_gpu

    # Multi-GPU: print summary (single-GPU already printed inline)
    if gpu_count > 1:
        for key, _, label in precisions + [("fp8", None, "FP8")]:
            r = aggregated.get(key)
            if r:
                method = f" [yellow](median, CV={r['cv_pct']}%)[/]" if r.get("tflops_method") == "median" else ""
                console.print(
                    f"  [bold green]{label}:[/] {r['cluster_tflops']} TFLOPS (cluster)  "
                    f"[dim]({r['tflops']} avg/GPU)[/]{method}"
                )

    # Summary table
    table = Table(title="\nGEMM Results", show_lines=True, border_style="cyan")
    table.add_column("Precision", style="bold")
    table.add_column("TFLOPS", justify="right", style="green")
    if gpu_count > 1:
        table.add_column("Cluster", justify="right", style="bold green")
    table.add_column("Median ms", justify="right", style="dim")
    table.add_column("Avg ms", justify="right", style="dim")
    table.add_column("CV%", justify="right")
    table.add_column("Method", style="dim")

    for key, _, label in precisions + [("fp8", None, "FP8")]:
        r = results.get(key)
        if r:
            cv = r.get("cv_pct", 0)
            cv_style = "red" if cv > 5.0 else "yellow" if cv > 2.0 else "green"
            method = r.get("tflops_method", "trimmed mean")
            row = [label, str(r["tflops"])]
            if gpu_count > 1:
                row.append(str(r["cluster_tflops"]))
            row.extend([
                f"{r['median_ms']:.3f}",
                f"{r['avg_ms']:.3f}",
                f"[{cv_style}]{cv}[/]",
                method,
            ])
            table.add_row(*row)
        else:
            empty = [""] * (6 if gpu_count == 1 else 7)
            empty[0] = label
            empty[1] = "[dim]skipped[/]"
            table.add_row(*empty)

    console.print(table)
    write_result("01_gemm.json", results)


if __name__ == "__main__":
    main()
