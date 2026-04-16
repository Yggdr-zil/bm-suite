#!/usr/bin/env python3
"""
CU Benchmark Suite — Thermal Warmup Gate
Runs sustained GEMM until GPU reaches thermal steady state.

Steady state criteria (ALL must be true):
  1. Temperature stays within a BAND (max-min < 10C) over the SUSTAIN window
  2. Temperature is NOT monotonically rising — the mean of the second half of the
     window must not exceed the mean of the first half by more than 2C
  3. Minimum warmup floor has elapsed (prevents cold-start false positives)

Fan speed changes on H100 SXM cause periodic dips of 3-5C — this is normal
oscillation and should NOT reset the gate. The band check accommodates this.
"""
import subprocess
import time
import os
import json

from rich.live import Live
from rich.table import Table
from rich.text import Text

from _common import RESULTS_DIR, write_result, console, print_header

PLATFORM = os.environ.get("PLATFORM", "nvidia")
MIN_WARMUP_SECS = int(os.environ.get("CU_WARMUP_MIN_SECS", "300"))    # 5 min floor
MAX_WARMUP_SECS = int(os.environ.get("CU_WARMUP_MAX_SECS", "900"))    # 15 min ceiling
STEADY_BAND = float(os.environ.get("CU_STEADY_BAND", "10.0"))         # max-min < 10C
STEADY_SUSTAIN = int(os.environ.get("CU_STEADY_SUSTAIN", "120"))       # hold for 120s
CLIMB_THRESHOLD = 2.0   # second-half mean can't exceed first-half mean by more than this


def get_temp():
    try:
        if PLATFORM == "nvidia":
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5
            ).strip()
            temps = [int(t.strip()) for t in out.split("\n") if t.strip()]
            return max(temps)
        else:
            try:
                out = subprocess.check_output(
                    ["amd-smi", "metric", "-t", "--json"],
                    text=True, timeout=5
                )
                data = json.loads(out)
                return max(gpu.get("temperature", {}).get("hotspot", 0) for gpu in data)
            except (FileNotFoundError, json.JSONDecodeError):
                out = subprocess.check_output(
                    ["rocm-smi", "--showtemp"], text=True, timeout=5
                )
                import re
                temps = re.findall(r"(\d+\.?\d*)", out)
                return max(float(t) for t in temps) if temps else 0
    except Exception:
        return 0


def check_steady(readings, sustain_window):
    """Check if the last `sustain_window` seconds are thermally stable.

    Returns (is_steady, band_delta, is_climbing, first_half_mean, second_half_mean)
    """
    if len(readings) < sustain_window:
        return False, 0, False, 0, 0

    window = [x["temp"] for x in readings[-sustain_window:]]
    band_delta = max(window) - min(window)

    # Band check: must stay within STEADY_BAND
    band_ok = band_delta <= STEADY_BAND

    # Climb check: split window in half, compare means
    mid = len(window) // 2
    first_half = window[:mid]
    second_half = window[mid:]
    first_mean = sum(first_half) / len(first_half)
    second_mean = sum(second_half) / len(second_half)
    is_climbing = (second_mean - first_mean) > CLIMB_THRESHOLD

    is_steady = band_ok and not is_climbing
    return is_steady, band_delta, is_climbing, first_mean, second_mean


def build_table(readings, steady_reached):
    table = Table(show_lines=False, padding=(0, 1), title="[bold cyan]Thermal Warmup[/]")
    table.add_column("sec", style="dim", width=5, justify="right")
    table.add_column("temp C", justify="right", width=7)
    table.add_column("", width=40)
    table.add_column("status", width=36)

    elapsed_total = readings[-1]["sec"] if readings else 0

    for r in readings[-20:]:
        temp = r["temp"]
        elapsed = r["sec"]

        temp_style = "red" if temp > 85 else "yellow" if temp > 75 else "green"
        bar_len = max(0, temp - 30)
        bar = Text("█" * min(bar_len, 40), style="red" if temp > 85 else "yellow" if temp > 75 else "blue")

        if elapsed_total < MIN_WARMUP_SECS:
            remaining = MIN_WARMUP_SECS - elapsed_total
            mins = remaining // 60
            secs = remaining % 60
            status = f"[cyan]SOAKING[/] [dim]({mins}m{secs:02d}s remaining)[/]"
        elif len(readings) < STEADY_SUSTAIN:
            status = f"[cyan]CALIBRATING[/] [dim](need {STEADY_SUSTAIN}s readings)[/]"
        else:
            is_steady, band_delta, is_climbing, fm, sm = check_steady(readings, STEADY_SUSTAIN)
            if is_steady:
                status = "[bold green]STEADY[/]"
            elif is_climbing:
                status = f"[red]CLIMBING[/] [dim]Δmean={sm-fm:+.1f}C band={band_delta:.0f}C[/]"
            else:
                status = f"[yellow]band={band_delta:.0f}C[/] [dim](need <{STEADY_BAND:.0f}C)[/]"

        table.add_row(str(elapsed), f"[{temp_style}]{temp}[/]", bar, status)

    target_info = f"[dim]Min: {MIN_WARMUP_SECS}s | Band: <{STEADY_BAND:.0f}C / {STEADY_SUSTAIN}s | No climb | Max: {MAX_WARMUP_SECS}s[/]"
    if steady_reached:
        table.caption = f"[bold green]Steady state reached![/]  {target_info}"
    else:
        table.caption = target_info

    return table


def main():
    import torch

    if not torch.cuda.is_available():
        console.print("[yellow]No CUDA device. Skipping warmup.[/]")
        return

    device_name = torch.cuda.get_device_name(0)
    print_header(f"Thermal Warmup — {device_name}",
                 f"Min: {MIN_WARMUP_SECS}s | Band: <{STEADY_BAND:.0f}C / {STEADY_SUSTAIN}s | No climb | Max: {MAX_WARMUP_SECS}s")

    device = torch.device("cuda")
    # Match the GEMM benchmark dimension so the heat load is identical
    M = N = K = int(os.environ.get("CU_GEMM_DIM", "8192"))
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    readings = []
    start_time = time.time()
    start_temp = get_temp()
    steady_reached = False

    console.print(f"[bold cyan]Start temp:[/] {start_temp}C\n")

    with Live(build_table(readings, False), refresh_per_second=2, console=console) as live:
        while time.time() - start_time < MAX_WARMUP_SECS:
            # Run GEMM for ~1 second with sync after each op
            t0 = time.time()
            while time.time() - t0 < 1.0:
                _ = torch.mm(A, B)
                torch.cuda.synchronize()

            temp = get_temp()
            elapsed = int(time.time() - start_time)
            readings.append({"sec": elapsed, "temp": temp})

            # Only check steady state AFTER minimum warmup period
            if elapsed >= MIN_WARMUP_SECS and len(readings) >= STEADY_SUSTAIN:
                is_steady, _, _, _, _ = check_steady(readings, STEADY_SUSTAIN)
                if is_steady:
                    steady_reached = True
                    live.update(build_table(readings, True))
                    break

            live.update(build_table(readings, False))

    del A, B
    torch.cuda.empty_cache()

    end_temp = get_temp()
    duration = round(time.time() - start_time, 1)

    console.print(f"\n[bold green]Done:[/] {start_temp}C -> {end_temp}C in {duration}s")
    if not steady_reached:
        console.print("[yellow]WARNING: Steady state not reached within time limit[/]")

    warmup_data = {
        "start_temp_c": start_temp,
        "end_temp_c": end_temp,
        "warmup_seconds": duration,
        "steady_state_reached": steady_reached,
        "steady_band_c": STEADY_BAND,
        "steady_sustain_s": STEADY_SUSTAIN,
        "readings": readings,
    }
    write_result("00_warmup.json", warmup_data)


if __name__ == "__main__":
    main()
