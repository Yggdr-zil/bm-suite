#!/usr/bin/env python3
"""
CU Benchmark Suite — Terminal Plot Suite
Renders benchmark results as ASCII/Unicode charts directly in the terminal.
Works over SSH, no display required. Uses plotext for charts, rich for tables.

Charts shown:
  1. Telemetry timeline  — GPU temp + clock over full run (with stage bands)
  2. GEMM TFLOPS         — bar chart per precision, all GPUs
  3. Memory Bandwidth    — bar chart all GPUs
  4. VRAM                — verified vs reported per GPU
  5. Summary panel       — key metrics + integrity SHA
"""
import json
import os
import sys

try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

from _common import RESULTS_DIR, console, print_header

REPORT_PATH = os.path.join(RESULTS_DIR, "benchmark_report.json")


def load_report():
    if not os.path.exists(REPORT_PATH):
        console.print(f"[red]Report not found: {REPORT_PATH}[/]")
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


# ─── Telemetry timeline ──────────────────────────────────────────────────────

def show_telemetry(report):
    rows = report.get("telemetry", {}).get("rows", [])
    if not rows:
        console.print("  [dim]No telemetry data.[/]")
        return

    def col(key):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(key, "") or ""))
            except ValueError:
                out.append(None)
        return out

    # Filter to GPU 0 rows if multi-GPU telemetry present
    gpu0_rows = [r for r in rows if str(r.get("gpu_idx", "0")).strip() == "0"]
    if not gpu0_rows:
        gpu0_rows = rows

    def colg(key):
        out = []
        for r in gpu0_rows:
            try:
                out.append(float(r.get(key, "") or ""))
            except ValueError:
                out.append(None)
        return out

    temps  = colg("temp_c")
    clocks = colg("clock_gpu_mhz")
    powers = colg("power_w")
    x      = list(range(len(gpu0_rows)))

    # Stage bands: list of (start_idx, end_idx, label) from cumulative sample counts
    profiles = report.get("telemetry", {}).get("thermal_profiles", {})
    stage_order = ["thermal_warmup", "sustained_clock", "lock_clocks", "gemm",
                   "membw", "vram", "interconnect", "inference"]
    bands = []
    cursor = 0
    for s in stage_order:
        p = profiles.get(s, {})
        sc = p.get("sample_count", 0)
        if sc:
            bands.append((cursor, cursor + sc, s))
            cursor += sc

    if not HAS_PLOTEXT:
        console.print("  [dim]plotext not installed — install with: pip install plotext[/]")
        _show_telemetry_table(report, gpu0_rows)
        return

    plt.clf()
    plt.theme("dark")

    # Temperature chart
    plt.subplots(3, 1)
    plt.subplot(1, 1)
    valid_t = [v for v in temps if v is not None]
    plt.plot(x, [v if v is not None else 0 for v in temps], color="red", label="Temp °C")
    plt.title("GPU 0 Temperature (°C)")
    plt.ylim(0, max(valid_t) * 1.15 if valid_t else 100)
    for start, end, label in bands:
        plt.vertical_line(start, color="gray")

    plt.subplot(2, 1)
    valid_c = [v for v in clocks if v is not None]
    plt.plot(x, [v if v is not None else 0 for v in clocks], color="cyan", label="Clock MHz")
    plt.title("GPU Clock (MHz)")
    plt.ylim(0, max(valid_c) * 1.1 if valid_c else 3000)
    for start, end, label in bands:
        plt.vertical_line(start, color="gray")

    plt.subplot(3, 1)
    valid_p = [v for v in powers if v is not None]
    plt.plot(x, [v if v is not None else 0 for v in powers], color="yellow", label="Power W")
    plt.title("Power (W)")
    plt.ylim(0, max(valid_p) * 1.1 if valid_p else 1000)
    for start, end, label in bands:
        plt.vertical_line(start, color="gray")

    plt.xlabel("Sample (1Hz)")
    plt.show()

    # Print stage legend
    if bands:
        console.print("  [dim]Vertical lines: " +
                      " | ".join(f"{s[2]}" for s in bands) + "[/]")


def _show_telemetry_table(report, rows):
    """Fallback: show min/max/mean per stage as table."""
    profiles = report.get("telemetry", {}).get("thermal_profiles", {})
    if not profiles:
        return
    table = Table(title="Thermal Profile per Stage", show_lines=True, border_style="cyan")
    table.add_column("Stage", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Temp min/max/mean °C", justify="right")
    table.add_column("Clock min/max MHz", justify="right")
    table.add_column("Throttled?")
    for stage, prof in profiles.items():
        t = prof.get("temp_c") or {}
        c = prof.get("clock_gpu_mhz") or {}
        throttled = "[red]YES[/]" if prof.get("throttled") else "[green]no[/]"
        temp_str = f"{t.get('min','?')}/{t.get('max','?')}/{t.get('mean','?')}" if t else "-"
        clk_str  = f"{c.get('min','?')}/{c.get('max','?')}" if c else "-"
        table.add_row(stage, str(prof.get("sample_count", "-")), temp_str, clk_str, throttled)
    console.print(table)


# ─── GEMM TFLOPS bar chart ───────────────────────────────────────────────────

def show_gemm(report):
    gemm = report.get("detailed", {}).get("gemm", {})
    if not gemm:
        return

    gpus = gemm.get("gpus", {})
    gpu_keys = sorted(gpus.keys())
    precisions = ["fp32", "fp16", "bf16", "fp8"]

    if not HAS_PLOTEXT:
        # Rich table fallback
        table = Table(title="GEMM TFLOPS", show_lines=True, border_style="cyan")
        table.add_column("GPU", style="bold")
        for p in precisions:
            table.add_column(p.upper(), justify="right", style="green")
        for k in gpu_keys:
            g = gpus[k]
            row = [k] + [str(g[p]["tflops"]) if g.get(p) else "-" for p in precisions]
            table.add_row(*row)
        console.print(table)
        return

    plt.clf()
    plt.theme("dark")

    labels = []
    values = []
    colors_map = {"fp32": "red", "fp16": "blue", "bf16": "green", "fp8": "orange"}

    # Grouped bar: GPU0/FP16, GPU1/FP16, ..., GPU0/BF16, ...
    for p in precisions:
        for k in gpu_keys:
            g = gpus.get(k, {})
            r = g.get(p)
            if r:
                labels.append(f"{k[-5:]}/{p.upper()}")
                values.append(r["tflops"])

    if values:
        plt.bar(labels, values, color="blue")
        plt.title("GEMM TFLOPS per GPU × Precision")
        plt.xlabel("GPU / Precision")
        plt.ylabel("TFLOPS")
        plt.show()


# ─── Memory Bandwidth ────────────────────────────────────────────────────────

def show_membw(report):
    membw = report.get("detailed", {}).get("membw", {})
    if not membw:
        return

    gpus = membw.get("gpus", {})
    gpu_keys = sorted(gpus.keys())

    if not HAS_PLOTEXT:
        table = Table(title="Memory Bandwidth", show_lines=True, border_style="cyan")
        table.add_column("GPU", style="bold")
        table.add_column("Clone 25% VRAM GB/s", justify="right", style="green")
        table.add_column("Clone 10% VRAM GB/s", justify="right")
        table.add_column("Mul 10% VRAM GB/s", justify="right")
        for k in gpu_keys:
            g = gpus[k]
            cl = g.get("clone_large") or {}
            cp = g.get("clone_primary") or {}
            mu = g.get("mul_primary") or {}
            table.add_row(k, str(cl.get("gbps", "-")),
                          str(cp.get("gbps", "-")), str(mu.get("gbps", "-")))
        console.print(table)
        return

    plt.clf()
    plt.theme("dark")

    labels, vals_large, vals_primary = [], [], []
    for k in gpu_keys:
        g = gpus.get(k, {})
        labels.append(k[-5:])
        vals_large.append((g.get("clone_large") or {}).get("gbps", 0))
        vals_primary.append((g.get("clone_primary") or {}).get("gbps", 0))

    plt.multiple_bar(labels, [vals_large, vals_primary],
                     label=["Clone 25% VRAM (primary)", "Clone 10% VRAM"],
                     color=["green", "cyan"])
    plt.title("Memory Bandwidth (GB/s) per GPU")
    plt.ylabel("GB/s")
    plt.show()


# ─── VRAM breakdown ──────────────────────────────────────────────────────────

def show_vram(report):
    vram = report.get("detailed", {}).get("vram", {})
    if not vram:
        return

    gpus = vram.get("gpus", {})
    gpu_keys = sorted(gpus.keys())

    if not HAS_PLOTEXT:
        table = Table(title="VRAM Verification", show_lines=True, border_style="cyan")
        table.add_column("GPU", style="bold")
        table.add_column("Verified GB", justify="right", style="bold green")
        table.add_column("Reported GB", justify="right")
        table.add_column("Delta GB", justify="right", style="yellow")
        table.add_column("Delta %", justify="right")
        for k in gpu_keys:
            g = gpus[k]
            delta_style = "red" if abs(g.get("delta_vs_reported_gb", 0)) > 0.5 else "yellow"
            table.add_row(k,
                f"{g.get('verified_total_gb', '-'):.4f}",
                f"{g.get('reported_total_gb', '-'):.2f}",
                f"[{delta_style}]{g.get('delta_vs_reported_gb', '-'):.4f}[/]",
                f"{g.get('delta_vs_reported_pct', '-'):.3f}%")
        console.print(table)
        return

    plt.clf()
    plt.theme("dark")

    labels = [k[-5:] for k in gpu_keys]
    reported = [gpus[k].get("reported_total_gb", 0) for k in gpu_keys]
    verified = [gpus[k].get("verified_total_gb", 0) for k in gpu_keys]

    plt.multiple_bar(labels, [reported, verified],
                     label=["Reported", "Verified"],
                     color=["blue", "green"])
    plt.title("VRAM per GPU — Verified vs Reported (GB)")
    plt.ylabel("GB")
    plt.show()


# ─── Summary panel ───────────────────────────────────────────────────────────

def show_summary(report):
    env   = report.get("environment", {})
    meas  = report.get("measured_specs", {})
    sha   = report.get("integrity_sha256", "")
    run_id = report.get("run_id", "?")
    ts    = report.get("timestamp", "?")

    gpu_name  = env.get("gpu_model", meas.get("device", "?"))
    gpu_count = env.get("gpu_count", 1)
    driver    = env.get("driver_version", "?")
    platform  = env.get("platform", "?")

    def v(key, unit=""):
        val = meas.get(key)
        return f"{val} {unit}".strip() if val is not None else "—"

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style="bold cyan", width=22)
    table.add_column(style="green")

    table.add_row("GPU", f"{gpu_name} × {gpu_count}")
    table.add_row("Platform", f"{platform} | Driver {driver}")
    table.add_row("Run ID", run_id)
    table.add_row("", "")
    table.add_row("FP32 (CUDA cores)", v("fp32_tflops", "TFLOPS"))
    table.add_row("FP16 (tensor)", v("fp16_tflops", "TFLOPS"))
    table.add_row("BF16 (tensor)", v("bf16_tflops", "TFLOPS"))
    table.add_row("FP8  (tensor)", v("fp8_tflops",  "TFLOPS"))
    table.add_row("Mem BW", v("membw_gbps", "GB/s"))
    table.add_row("VRAM usable", v("vram_usable_gb", "GB"))
    table.add_row("VRAM reported", v("vram_reported_gb", "GB"))
    table.add_row("Interconnect", v("interconnect_bw_gbps", "GB/s") +
                  f"  [dim]({meas.get('interconnect_source', '')})[/]")
    if meas.get("mtok_per_hour"):
        table.add_row("Inference", v("mtok_per_hour", "Mtok/hr"))
    table.add_row("", "")
    table.add_row("SHA256", f"[dim]{sha[:32]}...[/]")

    # Throttle warnings
    throttle_warnings = []
    for stage, prof in report.get("telemetry", {}).get("thermal_profiles", {}).items():
        if prof.get("throttled"):
            evts = prof.get("throttle_events", [])
            throttle_warnings.append(f"[red]⚠ {stage}: throttled ({len(evts)} events)[/]")

    panel_content = table
    console.print(Panel(table, title=f"[bold cyan]CU Benchmark Summary[/]",
                        border_style="cyan", padding=(1, 2)))

    if throttle_warnings:
        for w in throttle_warnings:
            console.print(f"  {w}")

    console.print(f"\n[dim]Report: {REPORT_PATH}[/]")
    console.print(f"[dim]Plots:  {os.path.join(RESULTS_DIR, 'plots', 'summary.html')}[/]")


# ─── Main ────────────────────────────────────────────────────────────────────

def wait_for_q():
    """Block until the user presses 'q'. Works over SSH (raw terminal mode)."""
    import termios
    import tty

    console.print(
        "\n[bold cyan]Results uploaded. Press [bold white][q][/] to exit.[/]",
        end="",
    )
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch.lower() == "q" or ch in ("\x03", "\x04"):  # q, Ctrl+C, Ctrl+D
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    console.print()  # newline after 'q'


def main():
    print_header("Terminal Visualization", REPORT_PATH)

    if not HAS_PLOTEXT:
        console.print("  [yellow]plotext not installed — showing table fallback.[/]")
        console.print("  [dim]pip install plotext[/]\n")

    report = load_report()
    if not report:
        sys.exit(1)

    console.rule("[bold cyan]Telemetry Timeline[/]")
    show_telemetry(report)

    console.rule("[bold cyan]GEMM Throughput[/]")
    show_gemm(report)

    console.rule("[bold cyan]Memory Bandwidth[/]")
    show_membw(report)

    console.rule("[bold cyan]VRAM Verification[/]")
    show_vram(report)

    console.rule("[bold cyan]Summary[/]")
    show_summary(report)

    # Data is already uploaded. Block here so results stay visible on screen.
    try:
        wait_for_q()
    except Exception:
        # Non-interactive (piped, CI, etc.) — just exit cleanly
        pass


if __name__ == "__main__":
    main()
