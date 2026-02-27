#!/usr/bin/env python3
"""CU Benchmark Suite — Report Compiler

Merges all benchmark results into a single report with integrity hash.
Can be run standalone after individual benchmarks, or as the final step in run_all.sh.
"""
import csv
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


def _parse_ts(ts_str):
    """Parse nvidia-smi or ISO-8601 timestamp to datetime (UTC)."""
    ts_str = ts_str.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def load_telemetry():
    """Load telemetry.csv → list of row dicts. Returns [] if missing."""
    path = os.path.join(RESULTS_DIR, "telemetry.csv")
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys and values (nvidia-smi pads with spaces)
                clean = {k.strip(): v.strip() for k, v in row.items() if k}
                ts = _parse_ts(clean.get("timestamp", ""))
                if ts:
                    clean["_ts"] = ts
                    rows.append(clean)
    except Exception:
        pass
    return rows


def load_stage_events():
    """Load stage_events.csv → dict of {stage_name: {start_ts, end_ts, exit_code}}."""
    path = os.path.join(RESULTS_DIR, "stage_events.csv")
    if not os.path.exists(path):
        return {}
    stages = {}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("stage_name", "").strip()
                event = row.get("event", "").strip()
                ts = _parse_ts(row.get("timestamp_utc", ""))
                if not name or not ts:
                    continue
                if name not in stages:
                    stages[name] = {}
                if event == "start":
                    stages[name]["start_ts"] = ts
                elif event == "end":
                    stages[name]["end_ts"] = ts
                    stages[name]["exit_code"] = int(row.get("exit_code", "0") or "0")
    except Exception:
        pass
    return stages


def compute_thermal_profiles(telemetry_rows, stage_events):
    """For each stage, extract min/max/mean of temp/power/clock from telemetry rows
    that fall within [start_ts, end_ts]. Also detect throttle events."""
    profiles = {}
    MEASUREMENT_STAGES = {"gemm", "membw", "vram", "interconnect", "inference"}

    for stage_name, bounds in stage_events.items():
        start_ts = bounds.get("start_ts")
        end_ts = bounds.get("end_ts")
        if not start_ts or not end_ts:
            continue

        window = [r for r in telemetry_rows if r.get("_ts") and start_ts <= r["_ts"] <= end_ts]
        if not window:
            profiles[stage_name] = {"sample_count": 0}
            continue

        def col_floats(key):
            vals = []
            for r in window:
                try:
                    vals.append(float(r.get(key, "") or ""))
                except ValueError:
                    pass
            return vals

        def stats(vals):
            if not vals:
                return None
            return {"min": round(min(vals), 1), "max": round(max(vals), 1),
                    "mean": round(sum(vals) / len(vals), 1)}

        profile = {
            "sample_count": len(window),
            "duration_s": round((end_ts - start_ts).total_seconds(), 1),
            "temp_c": stats(col_floats("temp_c")),
            "power_w": stats(col_floats("power_w")),
            "clock_gpu_mhz": stats(col_floats("clock_gpu_mhz")),
        }

        # Throttle detection — only flag during measurement stages
        if stage_name in MEASUREMENT_STAGES:
            throttle_events = []
            for r in window:
                reason = r.get("throttle_reasons", "").strip()
                if reason and reason.lower() not in ("no active throttle reasons", "0x0000000000000000", ""):
                    throttle_events.append({
                        "ts": r["_ts"].isoformat(),
                        "reason": reason,
                        "clock_mhz": r.get("clock_gpu_mhz", ""),
                    })
            if throttle_events:
                profile["throttle_events"] = throttle_events
                profile["throttled"] = True
            else:
                profile["throttled"] = False

        profiles[stage_name] = profile

    return profiles


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

    # Load telemetry + stage events
    telemetry_rows = load_telemetry()
    stage_events = load_stage_events()
    thermal_profiles = compute_thermal_profiles(telemetry_rows, stage_events)

    telemetry_summary = {
        "row_count": len(telemetry_rows),
        "stages_tracked": len(stage_events),
        "telemetry_loaded": len(telemetry_rows) > 0,
    }
    if telemetry_rows:
        console.print(f"  Telemetry: [green]{len(telemetry_rows)} rows[/] | "
                      f"{len(stage_events)} stages | {len(thermal_profiles)} profiles")
    else:
        console.print("  [dim]Telemetry: not found (telemetry.csv missing)[/]")

    # ─── Extract primary measurements ───
    measured = {}

    for prec in ["fp32", "fp16", "bf16", "fp8"]:
        val = gemm.get(prec)
        measured[f"{prec}_tflops"] = val.get("tflops") if val and isinstance(val, dict) else None

    bw = membw.get("clone_large") or membw.get("clone_primary") or {}
    measured["membw_gbps"] = bw.get("gbps")

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
    # Serialize telemetry rows (strip internal _ts datetime objects)
    telemetry_serializable = [
        {k: v for k, v in r.items() if k != "_ts"}
        for r in telemetry_rows
    ]
    # Serialize thermal_profiles (convert any datetime objects to ISO strings)
    def _serialize_profiles(profiles):
        result = {}
        for stage, prof in profiles.items():
            clean_prof = {}
            for k, v in prof.items():
                if k == "throttle_events":
                    clean_prof[k] = v  # already serialized inside compute_thermal_profiles
                else:
                    clean_prof[k] = v
            result[stage] = clean_prof
        return result

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
        "telemetry": {
            "summary": telemetry_summary,
            "stage_events": {k: {
                "start_ts": v.get("start_ts").isoformat() if v.get("start_ts") else None,
                "end_ts": v.get("end_ts").isoformat() if v.get("end_ts") else None,
                "exit_code": v.get("exit_code"),
            } for k, v in stage_events.items()},
            "thermal_profiles": _serialize_profiles(thermal_profiles),
            "rows": telemetry_serializable,
        },
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
