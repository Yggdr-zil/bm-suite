#!/usr/bin/env python3
"""
CU Benchmark Suite — Auto-Plot Suite
Reads benchmark_report.json and generates PNG charts + HTML summary.

Charts produced:
  telemetry_overview.png  — 3-panel: temp / power / GPU clock (with stage boundaries)
  gemm_tflops.png         — FP32/FP16/BF16/FP8 TFLOPS bar chart
  gemm_drift_*.png        — per-precision timing drift time-series
  membw.png               — memory bandwidth bars
  interconnect.png        — NVLink / PCIe sweep curve (if available)
  vram.png                — VRAM per-GPU breakdown
  summary.html            — all charts in one page

Runs headless (Agg backend) — safe for Docker.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from _common import RESULTS_DIR, console, print_header


REPORT_PATH = os.path.join(RESULTS_DIR, "benchmark_report.json")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

STAGE_COLORS = {
    "thermal_warmup": "#FF9800",
    "sustained_clock": "#9C27B0",
    "lock_clocks": "#607D8B",
    "gemm": "#2196F3",
    "membw": "#4CAF50",
    "vram": "#00BCD4",
    "interconnect": "#FF5722",
    "inference": "#E91E63",
    "report": "#9E9E9E",
}
PRECISION_COLORS = {"FP32": "#EF5350", "FP16": "#42A5F5", "BF16": "#66BB6A", "FP8": "#FFA726"}


def load_report():
    if not os.path.exists(REPORT_PATH):
        console.print(f"[red]benchmark_report.json not found at {REPORT_PATH}[/]")
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return path


# ─── Telemetry overview ───────────────────────────────────────────────────────

def plot_telemetry(report):
    rows = report.get("telemetry", {}).get("rows", [])
    if not rows:
        return None

    def col(key):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(key, "") or ""))
            except ValueError:
                out.append(float("nan"))
        return out

    temps   = col("temp_c")
    powers  = col("power_w")
    clocks  = col("clock_gpu_mhz")
    x       = list(range(len(rows)))

    # Stage boundary x-positions (map stage start/end ts → row index by position)
    profiles = report.get("telemetry", {}).get("thermal_profiles", {})
    stage_events_raw = report.get("telemetry", {}).get("stage_events", {})

    # Build a simple ts → index map using sorted row positions
    # (rows are in chronological order from 1Hz telemetry)
    n_rows = len(rows)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor="#1a1a2e",
                              gridspec_kw={"hspace": 0.08})
    fig.suptitle("Telemetry Overview", color="white", fontsize=14, y=0.98)

    datasets = [
        (axes[0], temps,  "Temp (°C)",        "#FF7043", (0, 100)),
        (axes[1], powers, "Power (W)",         "#FFA726", None),
        (axes[2], clocks, "GPU Clock (MHz)",   "#42A5F5", None),
    ]

    for ax, data, ylabel, color, ylim in datasets:
        ax.set_facecolor("#0d0d1a")
        ax.plot(x, data, color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#333", linewidth=0.5)

    # Add stage boundary lines on all panels
    legend_patches = []
    drawn_stages = set()
    for stage_name, prof in profiles.items():
        dur = prof.get("duration_s")
        if not dur:
            continue
        # Approximate start index from stage_events (proportional to total rows)
        se = stage_events_raw.get(stage_name, {})
        # We use relative position within the run — map via sample count if available
        sample_count = prof.get("sample_count", 0)
        # Find position by scanning for a stage with known sample_count → row span
        # Fallback: draw boundary at midpoint with duration span
        color = STAGE_COLORS.get(stage_name, "#888")
        for ax, _, _, _, _ in datasets:
            ax.axvspan(0, 0, alpha=0)  # placeholder; real boundaries drawn below

    # Draw boundaries using cumulative sample counts per stage
    stage_order = ["thermal_warmup", "sustained_clock", "lock_clocks", "gemm",
                   "membw", "vram", "interconnect", "inference", "report"]
    cursor = 0
    for stage_name in stage_order:
        prof = profiles.get(stage_name)
        if not prof or not prof.get("sample_count"):
            continue
        span = prof["sample_count"]
        color = STAGE_COLORS.get(stage_name, "#888888")
        for ax, _, _, _, _ in datasets:
            ax.axvspan(cursor, cursor + span, alpha=0.10, color=color, linewidth=0)
            ax.axvline(cursor, color=color, linewidth=0.7, alpha=0.6)
        if stage_name not in drawn_stages:
            legend_patches.append(mpatches.Patch(color=color, alpha=0.5, label=stage_name))
            drawn_stages.add(stage_name)
        cursor += span

    axes[0].legend(handles=legend_patches, loc="upper right",
                   fontsize=7, facecolor="#1a1a2e", labelcolor="white", framealpha=0.7)
    axes[2].set_xlabel("Telemetry sample (1Hz)", color="white", fontsize=9)

    plt.savefig(os.path.join(PLOTS_DIR, "telemetry_overview.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return "telemetry_overview.png"


# ─── GEMM charts ─────────────────────────────────────────────────────────────

def plot_gemm(report):
    gemm = report.get("detailed", {}).get("gemm", {})
    if not gemm:
        return []

    produced = []

    # Bar chart: TFLOPS per precision
    precisions = ["fp32", "fp16", "bf16", "fp8"]
    labels, values, colors = [], [], []
    for p in precisions:
        r = gemm.get(p)
        if r and r.get("tflops"):
            labels.append(p.upper())
            values.append(r["tflops"])
            colors.append(PRECISION_COLORS.get(p.upper(), "#888"))

    if values:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#0d0d1a")
        bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.02,
                    f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=10)
        ax.set_ylabel("TFLOPS", color="white")
        ax.set_title(f"GEMM Throughput — {gemm.get('device', '?')}", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.grid(axis="y", color="#333", linewidth=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        name = "gemm_tflops.png"
        plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()
        produced.append(name)

    # Drift time-series per precision
    for p in precisions:
        r = gemm.get(p)
        if not r or not r.get("sampled_times_ms"):
            continue
        samples = r["sampled_times_ms"]
        x = list(range(len(samples)))
        color = PRECISION_COLORS.get(p.upper(), "#888")
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1a1a2e")
        ax.set_facecolor("#0d0d1a")
        ax.plot(x, samples, color=color, linewidth=0.9)
        ax.axhline(r.get("median_ms", 0), color="#aaa", linewidth=0.8, linestyle="--",
                   label=f"median {r['median_ms']:.3f}ms")
        ax.set_title(f"GEMM {p.upper()} timing drift (sampled)", color="white", fontsize=11)
        ax.set_xlabel("Sample index (1 per 100 iters)", color="white", fontsize=9)
        ax.set_ylabel("ms / iter", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        ax.grid(color="#333", linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        name = f"gemm_drift_{p}.png"
        plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()
        produced.append(name)

    return produced


# ─── Memory bandwidth ─────────────────────────────────────────────────────────

def plot_membw(report):
    membw = report.get("detailed", {}).get("membw", {})
    if not membw:
        return None

    tests = []
    for key in ["clone_primary", "clone_large", "mul_primary"]:
        r = membw.get(key)
        if r and r.get("gbps"):
            tests.append((key.replace("_", "\n"), r["gbps"], r.get("sampled_gbps")))

    if not tests:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1a1a2e",
                              gridspec_kw={"width_ratios": [1, 2]})

    # Bar chart
    ax = axes[0]
    ax.set_facecolor("#0d0d1a")
    labels = [t[0] for t in tests]
    vals   = [t[1] for t in tests]
    bars = ax.bar(labels, vals, color=["#4CAF50", "#8BC34A", "#CDDC39"], width=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02,
                f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=10)
    ax.set_ylabel("GB/s", color="white")
    ax.set_title("Memory Bandwidth", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.grid(axis="y", color="#333", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Drift series for clone_large
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d1a")
    for label, gbps, sampled in tests:
        if sampled:
            ax2.plot(sampled, label=f"{label.replace(chr(10), ' ')} ({gbps:.0f} GB/s)",
                     linewidth=0.9)
    ax2.set_title("Bandwidth drift (sampled)", color="white", fontsize=11)
    ax2.set_xlabel("Sample index", color="white", fontsize=9)
    ax2.set_ylabel("GB/s", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=7)
    ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax2.grid(color="#333", linewidth=0.4)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    fig.suptitle(f"Memory BW — {membw.get('device', '?')}", color="white", fontsize=12)
    name = "membw.png"
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return name


# ─── VRAM breakdown ───────────────────────────────────────────────────────────

def plot_vram(report):
    vram = report.get("detailed", {}).get("vram", {})
    if not vram:
        return None

    gpus = vram.get("gpus", {})
    if not gpus:
        return None

    labels, verified, reported = [], [], []
    for gpu_id, info in sorted(gpus.items()):
        labels.append(f"GPU {gpu_id}")
        verified.append(info.get("verified_total_gb", 0) or 0)
        reported.append(info.get("reported_total_gb", 0) or 0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4), facecolor="#1a1a2e")
    ax.set_facecolor("#0d0d1a")
    w = 0.35
    bars_r = ax.bar(x - w/2, reported, w, label="Reported", color="#42A5F5", zorder=3)
    bars_v = ax.bar(x + w/2, verified, w, label="Verified", color="#66BB6A", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white")
    ax.set_ylabel("GB", color="white")
    ax.set_title(f"VRAM — total verified: {vram.get('total_verified_gb', '?')} GB", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.grid(axis="y", color="#333", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    name = "vram.png"
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return name


# ─── Interconnect ─────────────────────────────────────────────────────────────

def plot_interconnect(report):
    ic = report.get("detailed", {}).get("interconnect", {})
    if not ic or ic.get("skipped"):
        return None

    raw_output = ic.get("raw_output", "")
    # Try to parse sweep data: look for lines with "  float  " and bandwidth values
    sweep_sizes_gb = []
    sweep_busbw    = []
    for line in (raw_output or "").splitlines():
        parts = line.split()
        # nccl-tests output: size(B) count type redop root time algbw busbw
        if len(parts) >= 8:
            try:
                size_b = int(parts[0])
                busbw  = float(parts[6])  # busbw column
                sweep_sizes_gb.append(size_b / 1e9)
                sweep_busbw.append(busbw)
            except (ValueError, IndexError):
                pass

    if not sweep_busbw:
        # Fall back to summary bar
        peak = ic.get("peak_busbw_gbps")
        if not peak:
            return None
        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e")
        ax.set_facecolor("#0d0d1a")
        ax.bar(["Peak NVLink busbw"], [peak], color="#FF5722", width=0.4)
        ax.set_ylabel("GB/s", color="white")
        ax.set_title("Interconnect bandwidth", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
    else:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#0d0d1a")
        ax.plot(sweep_sizes_gb, sweep_busbw, color="#FF5722", linewidth=1.2, marker="o",
                markersize=3)
        ax.set_xlabel("Message size (GB)", color="white", fontsize=9)
        ax.set_ylabel("Bus bandwidth (GB/s)", color="white", fontsize=9)
        ax.set_title("NVLink all_reduce sweep", color="white", fontsize=12)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(color="#333", linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    name = "interconnect.png"
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return name


# ─── HTML summary ────────────────────────────────────────────────────────────

def write_html(report, charts):
    gpu = report.get("environment", {}).get("gpu_model", "?")
    run_id = report.get("run_id", "?")
    ts = report.get("timestamp", "?")

    chart_tags = "\n".join(
        f'<div class="chart"><img src="{c}" alt="{c}"></div>'
        for c in charts if c
    )

    meas = report.get("measured_specs", {})

    rows_html = ""
    for key, label, unit in [
        ("fp16_tflops", "FP16", "TFLOPS"),
        ("bf16_tflops", "BF16", "TFLOPS"),
        ("fp32_tflops", "FP32", "TFLOPS"),
        ("fp8_tflops",  "FP8",  "TFLOPS"),
        ("membw_gbps",  "Mem BW", "GB/s"),
        ("vram_usable_gb", "VRAM Usable", "GB"),
        ("interconnect_bw_gbps", "Interconnect", "GB/s"),
        ("mtok_per_hour", "Inference", "Mtok/hr"),
    ]:
        val = meas.get(key)
        rows_html += f"<tr><td>{label}</td><td>{val if val is not None else '—'}</td><td>{unit}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CU Benchmark — {gpu}</title>
<style>
  body {{ background:#1a1a2e; color:#eee; font-family:monospace; margin:20px; }}
  h1 {{ color:#42A5F5; }} h2 {{ color:#aaa; font-size:1em; }}
  table {{ border-collapse:collapse; margin-bottom:20px; }}
  td, th {{ padding:6px 14px; border:1px solid #333; }}
  th {{ background:#0d0d1a; color:#42A5F5; }}
  .chart {{ margin:20px 0; }}
  .chart img {{ max-width:100%; border:1px solid #333; border-radius:4px; }}
</style></head>
<body>
<h1>CU Benchmark Report — {gpu}</h1>
<h2>Run ID: {run_id} &nbsp;|&nbsp; {ts}</h2>
<table><tr><th>Metric</th><th>Value</th><th>Unit</th></tr>
{rows_html}
</table>
{chart_tags}
</body></html>"""

    path = os.path.join(PLOTS_DIR, "summary.html")
    with open(path, "w") as f:
        f.write(html)
    return "summary.html"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print_header("Plot Suite", f"Reading {REPORT_PATH}")

    report = load_report()
    if not report:
        sys.exit(1)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    charts = []
    charts.append(plot_telemetry(report))
    charts.extend(plot_gemm(report))
    charts.append(plot_membw(report))
    charts.append(plot_vram(report))
    charts.append(plot_interconnect(report))

    produced = [c for c in charts if c]
    html_name = write_html(report, produced)
    produced.append(html_name)

    console.print(f"\n  [green]{len(produced)} outputs[/] → {PLOTS_DIR}/")
    for name in produced:
        console.print(f"  [dim]{name}[/]")


if __name__ == "__main__":
    main()
