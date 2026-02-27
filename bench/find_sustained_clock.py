#!/usr/bin/env python3
"""
CU Benchmark Suite — Find Sustained Clock Frequency
Runs GEMM for up to 180 seconds, shows live clock/temp/power to find steady-state frequency.
Useful for Blackwell consumer GPUs where Default Applications Clocks is deprecated.

Can be run standalone or called from preflight.py.
"""
import os
import time
import subprocess

import torch
from rich.live import Live
from rich.table import Table
from rich.text import Text

from _common import RESULTS_DIR, write_result, console, print_header

DURATION = int(os.environ.get("CU_SUSTAINED_CLK_DURATION", "180"))
STEADY_WINDOW = 30       # seconds of clock readings to check for stability
STEADY_DELTA_MHZ = 30    # max MHz spread to consider "steady"
MIN_RUNTIME = 60         # always run at least this long (thermal soak)


def find_sustained_clock(duration=DURATION, dim=None, show_header=True):
    """Run GEMM for `duration` seconds, return sustained clock info.

    Returns dict with:
        sustained_clock_mhz: int
        clock_range_mhz: [min, max]
        final_temp_c: int
        final_power_w: float
        readings: list of {sec, clock, temp, power}
    """
    if dim is None:
        dim = int(os.environ.get("CU_GEMM_DIM", "8192"))

    if show_header:
        print_header(
            "Sustained Clock Finder",
            f"GEMM {dim}x{dim} for {duration}s — sampling every 1s"
        )

    A = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
    B = torch.randn(dim, dim, device="cuda", dtype=torch.float16)

    readings = []
    steady_reached = False

    def check_steady():
        """Check if clock has stabilized over the steady window."""
        if len(readings) < STEADY_WINDOW:
            return False, 0, 0
        window = [r["clock"] for r in readings[-STEADY_WINDOW:]]
        delta = max(window) - min(window)
        avg = int(sum(window) / len(window))
        return delta <= STEADY_DELTA_MHZ, delta, avg

    def build_table():
        table = Table(title="GPU Clock / Temp / Power", show_lines=False, padding=(0, 1))
        table.add_column("sec", style="dim", width=4, justify="right")
        table.add_column("clock MHz", justify="right", width=10)
        table.add_column("temp C", justify="right", width=7)
        table.add_column("power W", justify="right", width=8)
        table.add_column("", width=20)
        table.add_column("status", width=20)

        for r in readings[-25:]:
            clk = r["clock"]
            temp = r["temp"]

            clk_style = "green" if steady_reached else "yellow" if len(readings) > MIN_RUNTIME else "cyan"
            temp_style = "red" if temp > 85 else "yellow" if temp > 75 else "green"

            bar_len = int(clk / 100)
            bar = Text("█" * min(bar_len, 20), style="blue")

            if len(readings) < MIN_RUNTIME:
                remaining = MIN_RUNTIME - len(readings)
                status = f"[cyan]SOAKING[/] [dim]({remaining}s)[/]"
            elif len(readings) < STEADY_WINDOW:
                remaining = STEADY_WINDOW - len(readings)
                status = f"[cyan]CALIBRATING[/] [dim]({remaining}s)[/]"
            else:
                is_steady, delta, avg = check_steady()
                if is_steady:
                    status = f"[bold green]STEADY[/] [dim]{avg} MHz[/]"
                else:
                    status = f"[yellow]delta={delta} MHz[/]"

            table.add_row(
                str(r["sec"]),
                f"[{clk_style}]{clk}[/]",
                f"[{temp_style}]{temp}[/]",
                f"{r['power']:.1f}",
                bar,
                status,
            )

        if len(readings) >= STEADY_WINDOW:
            _, delta, avg = check_steady()
            target = f"[dim]Target: delta < {STEADY_DELTA_MHZ} MHz over {STEADY_WINDOW}s | Max: {duration}s[/]"
            if steady_reached:
                table.caption = f"[bold green]Clock stabilized: {avg} MHz[/]  {target}"
            else:
                table.caption = f"[bold]Avg: {avg} MHz  (delta: {delta} MHz)[/]  {target}"

        return table

    with Live(build_table(), refresh_per_second=2, console=console) as live:
        for i in range(duration):
            t0 = time.time()
            while time.time() - t0 < 1:
                torch.mm(A, B)
                torch.cuda.synchronize()

            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=clocks.current.graphics,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], text=True).strip()

            parts = [p.strip() for p in out.split(",")]
            clk = int(parts[0])
            temp = int(parts[1])
            power = float(parts[2])

            readings.append({"sec": i + 1, "clock": clk, "temp": temp, "power": power})
            live.update(build_table())

            # Check for early exit: past minimum runtime AND clock is steady
            if i + 1 >= MIN_RUNTIME:
                is_steady, delta, avg_clk = check_steady()
                if is_steady:
                    steady_reached = True
                    live.update(build_table())
                    break

    del A, B
    torch.cuda.empty_cache()

    # Use the steady window for the final average
    window = [r["clock"] for r in readings[-STEADY_WINDOW:]]
    avg = int(sum(window) / len(window))
    mn, mx = min(window), max(window)

    result = {
        "sustained_clock_mhz": avg,
        "steady_state_reached": steady_reached,
        "clock_range_mhz": [mn, mx],
        "final_temp_c": readings[-1]["temp"],
        "final_power_w": readings[-1]["power"],
        "actual_duration_seconds": len(readings),
        "gemm_dim": dim,
        "readings": readings,
    }

    console.print(f"\n[bold green]Sustained clock:[/] {avg} MHz (range: {mn}-{mx}, delta: {mx-mn} MHz)")
    console.print(f"[bold green]Final temp:[/] {readings[-1]['temp']}C")
    console.print(f"[bold green]Power draw:[/] {readings[-1]['power']:.1f}W")
    console.print(f"[bold green]Duration:[/] {len(readings)}s  {'(steady)' if steady_reached else '(max time reached)'}")
    if not steady_reached:
        console.print(f"[yellow]WARNING: Clock did not stabilize within {duration}s (delta: {mx-mn} MHz > {STEADY_DELTA_MHZ} MHz target)[/]")

    return result


def main():
    result = find_sustained_clock()
    avg = result["sustained_clock_mhz"]

    console.print(f"\n[dim]Lock recommendation:[/] [bold]sudo nvidia-smi -lgc {avg},{avg}[/]")
    console.print(f"[dim]Or set:[/] [bold]export CU_GPU_CLOCK={avg}[/]")

    write_result("00_sustained_clock.json", result)


if __name__ == "__main__":
    main()
