#!/usr/bin/env python3
"""
CU Benchmark Suite — GEMM Benchmark
Measures sustained floating-point throughput at FP32, FP16, BF16, FP8.

Method: Square matrix multiply (M=N=K=8192) saturates tensor cores.
TFLOPS = 2 * M * N * K / time / 1e12
"""
import torch
import time
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


def bench_gemm(M, N, K, dtype, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run GEMM benchmark for a given precision."""
    device = torch.device("cuda")

    # Check VRAM budget before allocating (2 matrices + 1 output)
    elem_bytes = torch.finfo(dtype).bits // 8
    needed = (M * K + K * N + M * N) * elem_bytes
    if not check_vram_ok(needed, label):
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


def bench_gemm_fp8(M, N, K, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """FP8 GEMM uses torch._scaled_mm (Hopper+/Blackwell only)."""
    device = torch.device("cuda")

    # FP8 needs intermediate FP16 + FP8 copies — budget ~4 bytes/element
    needed = (M * K + K * N + M * N) * 4
    if not check_vram_ok(needed, "FP8"):
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


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device. Skipping GEMM benchmark.[/]")
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    print_header(
        f"{device_name}",
        f"Matrix: {MATRIX_DIM}x{MATRIX_DIM} | Warmup: {WARMUP_ITERS} | Iters: {BENCH_ITERS}",
    )

    M = N = K = MATRIX_DIM
    results = {"device": device_name, "matrix_dim": M}

    precisions = [
        ("fp32", torch.float32, "FP32"),
        ("fp16", torch.float16, "FP16"),
        ("bf16", torch.bfloat16, "BF16"),
    ]

    for key, dtype, label in precisions:
        r = bench_gemm(M, N, K, dtype, label)
        results[key] = r
        if r:
            method = f" [yellow](median, CV={r['cv_pct']}%)[/]" if r.get("tflops_method") == "median" else ""
            console.print(f"  [bold green]{label}:[/] {r['tflops']} TFLOPS  [dim]({r['avg_ms']:.2f} ms avg)[/]{method}")

    # FP8
    r = bench_gemm_fp8(M, N, K)
    results["fp8"] = r
    if r:
        method = f" [yellow](median, CV={r['cv_pct']}%)[/]" if r.get("tflops_method") == "median" else ""
        console.print(f"  [bold green]FP8:[/]  {r['tflops']} TFLOPS  [dim]({r['avg_ms']:.2f} ms avg)[/]{method}")

    # Summary table
    table = Table(title="\nGEMM Results", show_lines=True, border_style="cyan")
    table.add_column("Precision", style="bold")
    table.add_column("TFLOPS", justify="right", style="green")
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
            table.add_row(label, str(r["tflops"]), f"{r['median_ms']:.3f}",
                          f"{r['avg_ms']:.3f}", f"[{cv_style}]{cv}[/]", method)
        else:
            table.add_row(label, "[dim]skipped[/]", "", "", "", "")

    console.print(table)
    write_result("01_gemm.json", results)


if __name__ == "__main__":
    main()
