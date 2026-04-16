#!/usr/bin/env python3
"""CU Benchmark Suite — Report Compiler

Merges all benchmark results into a single report with integrity hash.
Can be run standalone after individual benchmarks, or as the final step in run_all.sh.
"""
import json
import hashlib
import os
import sys
from datetime import datetime, timezone
from glob import glob

from rich.table import Table
from rich.panel import Panel

from _common import RESULTS_DIR, RUN_ID, write_result, console, print_header, get_run_meta


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Strip _meta from sub-results (it'll be in the report-level _meta)
        data.pop("_meta", None)
        return data
    return None


def main():
    print_header("Report Compiler", f"Scanning: {RESULTS_DIR}")

    env = load_json("00_environment.json") or {}
    warmup = load_json("00_warmup.json") or {}
    gemm = load_json("01_gemm.json") or {}
    membw = load_json("02_membw.json") or {}
    vram = load_json("03_vram.json") or {}
    interconnect = load_json("04_interconnect.json") or {}
    inference = load_json("05_inference.json") or {}

    # Show what was found
    found = []
    for name, data in [("environment", env), ("warmup", warmup), ("gemm", gemm),
                        ("membw", membw), ("vram", vram), ("interconnect", interconnect),
                        ("inference", inference)]:
        if data:
            found.append(f"[green]{name}[/]")
        else:
            found.append(f"[dim]{name} (missing)[/]")
    console.print(f"  Found: {', '.join(found)}")

    # ─── Extract primary measurements ───
    measured = {}

    # GPU count from environment
    measured["gpu_count"] = env.get("gpu_count", 1)

    # GEMM: prefer cluster_tflops (multi-GPU aggregate) over single-GPU tflops
    for prec in ["fp32", "fp16", "bf16", "fp8"]:
        val = gemm.get(prec)
        if val and isinstance(val, dict):
            measured[f"{prec}_tflops"] = val.get("cluster_tflops", val.get("tflops"))
        else:
            measured[f"{prec}_tflops"] = None

    # MemBW: prefer cluster_gbps (multi-GPU aggregate) over single-GPU gbps
    bw = membw.get("clone_large") or membw.get("clone_primary") or {}
    measured["membw_gbps"] = bw.get("cluster_gbps", bw.get("gbps"))

    vram_gpus = vram.get("gpus", {})
    measured["vram_usable_gb"] = vram.get("total_verified_gb")
    measured["vram_reported_gb"] = vram.get("total_reported_gb")
    measured["vram_per_gpu_gb"] = list(vram_gpus.values())[0].get("verified_total_gb") if vram_gpus else None

    if not interconnect.get("skipped"):
        measured["interconnect_bw_gbps"] = interconnect.get("peak_busbw_gbps")
        measured["interconnect_source"] = "measured"
    elif interconnect.get("reason") == "single_gpu":
        # Single-GPU: use PCIe 5.0 x16 proxy per methodology §3.7 (Stage 7 fallback)
        measured["interconnect_bw_gbps"] = 63.0
        measured["interconnect_source"] = "pcie5_x16_proxy"
    else:
        measured["interconnect_bw_gbps"] = None
        measured["interconnect_source"] = "unavailable"

    inf_batches = inference.get("batches", {})
    if inf_batches:
        best_batch = max(inf_batches.values(), key=lambda b: b.get("mtok_per_hour", 0))
        measured["mtok_per_hour"] = best_batch.get("mtok_per_hour")
        measured["tokens_per_second"] = best_batch.get("tokens_per_second")
    else:
        measured["mtok_per_hour"] = None
        measured["tokens_per_second"] = None

    # efficiency_factor — frozen constant for virtual cluster bridge (patent Claim 24)
    eff_cal = inference.get("efficiency_calibration") or {}
    measured["efficiency_factor"] = eff_cal.get("efficiency_factor")
    measured["efficiency_roofline_tok_s"] = eff_cal.get("roofline_single_tok_s")
    measured["inference_model_weight_gb"] = eff_cal.get("weight_bytes_gb")

    # ─── Build report ───
    report = {
        "cu_bench_version": "1.0.0",
        "run_id": RUN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "warmup": {
            "start_temp_c": warmup.get("start_temp_c"),
            "end_temp_c": warmup.get("end_temp_c"),
            "duration_seconds": warmup.get("warmup_seconds"),
        },
        "measured_specs": measured,
        "benchmarks_completed": [name for name, data in [
            ("warmup", warmup), ("gemm", gemm), ("membw", membw),
            ("vram", vram), ("interconnect", interconnect), ("inference", inference),
        ] if data and not data.get("skipped")],
        "detailed": {
            "gemm": gemm,
            "membw": membw,
            "vram": vram,
            "interconnect": interconnect,
            "inference": inference,
        },
    }

    # ─── Integrity seal ───
    # Hash covers canonical JSON of the report EXCLUDING integrity_sha256 itself.
    # Canonical form: json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    # Verification (any Python 3):
    #   data = json.load(open("benchmark_report.json"))
    #   stored = data.pop("integrity_sha256")
    #   canonical = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)
    #   assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == stored
    canonical = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report["integrity_sha256"] = sha
    report["integrity_method"] = (
        "sha256(json.dumps(report_without_integrity_fields, "
        "indent=2, sort_keys=True, ensure_ascii=True).encode('utf-8'))"
    )

    report_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=True)

    # ─── Pretty output ───
    gpu_name = env.get("gpu_model", gemm.get("device", "?"))
    gpu_count = env.get("gpu_count", 1)

    console.print()
    table = Table(title="CU Benchmark Report", show_lines=True, border_style="bold cyan",
                  title_style="bold cyan", padding=(0, 2))
    table.add_column("Metric", style="bold", width=16)
    table.add_column("Value", justify="right", style="green", width=14)
    table.add_column("Unit", style="dim", width=8)

    def v(key):
        """Format measured value, showing '-' for None/missing."""
        val = measured.get(key)
        return str(val) if val is not None else "-"

    rows = [
        ("GPU", f"{gpu_name} x{gpu_count}", ""),
        ("Platform", env.get("platform", "?"), ""),
        ("Driver", env.get("driver_version", "?"), ""),
        ("PyTorch", env.get("pytorch_version", "?"), ""),
        ("", "", ""),
        ("FP32", v("fp32_tflops"), "TFLOPS"),
        ("FP16", v("fp16_tflops"), "TFLOPS"),
        ("BF16", v("bf16_tflops"), "TFLOPS"),
        ("FP8", v("fp8_tflops"), "TFLOPS"),
        ("Mem BW", v("membw_gbps"), "GB/s"),
        ("VRAM", v("vram_usable_gb"), "GB"),
        ("Interconnect", v("interconnect_bw_gbps"), "GB/s"),
        ("Inference", v("mtok_per_hour"), "Mtok/hr"),
        ("Eff. Factor", v("efficiency_factor"), ""),
    ]
    for metric, value, unit in rows:
        if metric == "":
            table.add_section()
        else:
            table.add_row(metric, value, unit)

    console.print(table)
    console.print(f"\n[dim]Run ID:  {RUN_ID}[/]")
    console.print(f"[dim]SHA256:  {sha[:32]}...[/]")
    console.print(f"[dim]Report:  {report_path}[/]")


if __name__ == "__main__":
    main()
