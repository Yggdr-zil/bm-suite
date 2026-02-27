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
BENCH_ITERS  = int(os.environ.get("CU_GEMM_ITERS",  "10000" if not os.environ.get("CU_QUICK") else "1000"))
# Drop the slowest N% of samples (cuBLAS auto-tuner tail)
TRIM_PCT = float(os.environ.get("CU_GEMM_TRIM_PCT", "5"))


def bench_gemm(M, N, K, dtype, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_idx=0):
    """Run GEMM benchmark for a given precision on a specific device.

    FP32 note: on Ampere+, allow_tf32 defaults to True — cuBLAS would silently
    route FP32 matmul through TF32 tensor cores (~15x faster than CUDA cores).
    We disable it for the FP32 pass so we measure true CUDA-core FP32 throughput
    (matching the methodology fp32_ref = 67 TFLOPS per H100 SXM).
    We also capture the TF32 result separately since it's useful context.
    """
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)

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

    # FP32 only: disable TF32 to force CUDA cores, not tensor cores.
    # On Ampere+, allow_tf32=True (default) would route FP32 through TF32 tensor
    # cores and produce ~989 TFLOPS instead of the ~67 TFLOPS CUDA-core baseline
    # the CU methodology fp32_ref is calibrated against.
    is_fp32 = (dtype == torch.float32)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if is_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False

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

    # Restore TF32 setting
    if is_fp32:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

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

    # Percentile histogram over full (untrimmed) sample set
    n = len(sorted_times)
    def pct(p): return round(sorted_times[int(n * p / 100)] * 1000, 3)
    timing_percentiles_ms = {
        "p1": pct(1), "p5": pct(5), "p10": pct(10), "p25": pct(25),
        "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95), "p99": pct(99),
    }

    # Sampled time-series: every (iters // 100)th sample → ~100 drift points
    sample_stride = max(1, iters // 100)
    sampled_times_ms = [round(times[i] * 1000, 3) for i in range(0, len(times), sample_stride)]

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
        "device_idx": device_idx,
        "timing_percentiles_ms": timing_percentiles_ms,
        "sampled_times_ms": sampled_times_ms,
    }
    if is_fp32:
        result["tf32_disabled"] = True
        result["note"] = "CUDA-core FP32 (allow_tf32=False). True baseline for CU fp32_ref."
    if cv_pct > 5.0:
        result["tflops_method"] = "median"
        result["tflops_trimmed_mean"] = round(tflops, 2)
    return result


def bench_gemm_fp8(M, N, K, warmup=WARMUP_ITERS, iters=BENCH_ITERS, device_idx=0):
    """FP8 GEMM uses torch._scaled_mm (Hopper+/Blackwell only)."""
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)

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

        n = len(sorted_times)
        def pct(p): return round(sorted_times[int(n * p / 100)] * 1000, 3)
        timing_percentiles_ms = {
            "p1": pct(1), "p5": pct(5), "p10": pct(10), "p25": pct(25),
            "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95), "p99": pct(99),
        }
        sample_stride = max(1, iters // 100)
        sampled_times_ms = [round(times[i] * 1000, 3) for i in range(0, len(times), sample_stride)]

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
            "device_idx": device_idx,
            "timing_percentiles_ms": timing_percentiles_ms,
            "sampled_times_ms": sampled_times_ms,
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


def _bench_one_gpu(device_idx, M, N, K):
    """Run all precisions on a single GPU device. Returns dict of precision results."""
    gpu_results = {}
    precisions = [
        ("fp32", torch.float32, "FP32"),
        ("fp16", torch.float16, "FP16"),
        ("bf16", torch.bfloat16, "BF16"),
    ]
    for key, dtype, label in precisions:
        r = bench_gemm(M, N, K, dtype, f"GPU{device_idx}/{label}", device_idx=device_idx)
        gpu_results[key] = r
        if r:
            method = f" [yellow](median)[/]" if r.get("tflops_method") == "median" else ""
            console.print(f"  [bold]GPU {device_idx}[/] [green]{label}:[/] {r['tflops']} TFLOPS{method}")
    r = bench_gemm_fp8(M, N, K, device_idx=device_idx)
    gpu_results["fp8"] = r
    if r:
        console.print(f"  [bold]GPU {device_idx}[/] [green]FP8: [/] {r['tflops']} TFLOPS")
    return gpu_results


def main():
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device. Skipping GEMM benchmark.[/]")
        sys.exit(0)

    gpu_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print_header(
        f"{device_name} × {gpu_count}",
        f"Matrix: {MATRIX_DIM}×{MATRIX_DIM} | Warmup: {WARMUP_ITERS} | Iters: {BENCH_ITERS}\n"
        f"FP32 measured with allow_tf32=False (CUDA cores, not tensor cores)",
    )

    M = N = K = MATRIX_DIM
    results = {
        "device": device_name,
        "gpu_count": gpu_count,
        "matrix_dim": M,
        "gpus": {},
    }

    for dev_idx in range(gpu_count):
        dev_name = torch.cuda.get_device_name(dev_idx)
        console.print(f"\n[bold cyan]── GPU {dev_idx}: {dev_name} ──[/]")
        gpu_res = _bench_one_gpu(dev_idx, M, N, K)
        results["gpus"][f"gpu_{dev_idx}"] = {"device_name": dev_name, **gpu_res}

    # Representative (GPU 0) floated to top level for backward compat with report.py
    gpu0 = results["gpus"].get("gpu_0", {})
    for prec in ["fp32", "fp16", "bf16", "fp8"]:
        results[prec] = gpu0.get(prec)

    # Summary table — one row per GPU × precision
    table = Table(title="\nGEMM Results (all GPUs)", show_lines=True, border_style="cyan")
    table.add_column("GPU", style="bold")
    table.add_column("FP32", justify="right", style="red")
    table.add_column("FP16", justify="right", style="green")
    table.add_column("BF16", justify="right", style="green")
    table.add_column("FP8", justify="right", style="yellow")
    table.add_column("CV%↑", justify="right", style="dim")

    for dev_idx in range(gpu_count):
        gr = results["gpus"].get(f"gpu_{dev_idx}", {})
        def tflops(k): return str(gr[k]["tflops"]) if gr.get(k) else "-"
        max_cv = max(
            (gr[k]["cv_pct"] for k in ["fp32","fp16","bf16","fp8"] if gr.get(k)),
            default=0
        )
        cv_style = "red" if max_cv > 5.0 else "yellow" if max_cv > 2.0 else "green"
        table.add_row(
            f"GPU {dev_idx}", tflops("fp32"), tflops("fp16"),
            tflops("bf16"), tflops("fp8"), f"[{cv_style}]{max_cv}[/]"
        )

    console.print(table)
    write_result("01_gemm.json", results)


if __name__ == "__main__":
    main()
