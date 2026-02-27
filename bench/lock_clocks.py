#!/usr/bin/env python3
"""
CU Benchmark Suite — Lock Clocks
Locks GPU clocks after thermal warmup and sustained clock detection.
Reads environment from 00_environment.json, optionally uses sustained clock result.
Updates 00_environment.json with final locked values.
"""
import json
import os
import subprocess
import sys

from _common import RESULTS_DIR, console, print_header, write_result


def nvidia_lock(args, label):
    """Try an nvidia-smi lock command, print result with rich."""
    try:
        subprocess.check_output(
            ["nvidia-smi"] + args, text=True, stderr=subprocess.STDOUT, timeout=10
        )
        console.print(f"  [green]{label}[/]")
        return True
    except subprocess.CalledProcessError as e:
        if "not supported" in (e.output or "").lower():
            console.print(f"  [dim]{label} — not supported on this GPU[/]")
        else:
            console.print(f"  [yellow]{label} — failed (need sudo?)[/]")
        return False
    except Exception:
        console.print(f"  [yellow]{label} — failed[/]")
        return False


def main():
    # Load current environment
    env_path = os.path.join(RESULTS_DIR, "00_environment.json")
    if not os.path.exists(env_path):
        console.print("[red]No 00_environment.json found. Run preflight.py first.[/]")
        sys.exit(1)

    with open(env_path) as f:
        env = json.load(f)
    env.pop("_meta", None)

    platform = env.get("platform", "nvidia")
    if platform != "nvidia":
        console.print("[dim]AMD clocks already locked in preflight.[/]")
        return

    gpu_clk = env.get("target_gpu_clock")
    mem_clk = env.get("target_mem_clock")
    clk_source = env.get("clock_source", "unknown")

    # Check for sustained clock result (written by find_sustained_clock.py)
    sustained_path = os.path.join(RESULTS_DIR, "00_sustained_clock.json")
    if gpu_clk is None and os.path.exists(sustained_path):
        with open(sustained_path) as f:
            sustained = json.load(f)
        sustained.pop("_meta", None)
        gpu_clk = sustained["sustained_clock_mhz"]
        mn, mx = sustained["clock_range_mhz"]
        steady = sustained.get("steady_state_reached", False)
        clk_source = f"sustained clock ({'steady' if steady else 'best avg'}, range {mn}-{mx} MHz)"

    print_header("Lock Clocks", f"GPU: {gpu_clk or 'unlocked'} MHz | Mem: {mem_clk or 'unlocked'} MHz")
    console.print(f"  [dim]Source: {clk_source}[/]\n")

    # ─── Lock GPU clock ───
    if gpu_clk:
        nvidia_lock(["-lgc", f"{gpu_clk},{gpu_clk}"], f"GPU clock locked: {gpu_clk} MHz")
        env["gpu_clock_locked_mhz"] = int(gpu_clk)
        env["clock_source"] = clk_source
    else:
        console.print("  [dim]GPU clock: unlocked (no target frequency)[/]")
        env["gpu_clock_locked_mhz"] = "unlocked"

    # ─── Lock mem clock ───
    if mem_clk:
        cc = env.get("compute_capability", "")
        if cc.startswith("9."):
            nvidia_lock(["--lock-memory-clocks-deferred", f"{mem_clk},{mem_clk}"],
                        f"Mem clock locked (deferred): {mem_clk} MHz")
        else:
            nvidia_lock(["-lmc", f"{mem_clk},{mem_clk}"], f"Mem clock locked: {mem_clk} MHz")
        env["mem_clock_locked_mhz"] = int(mem_clk)
    else:
        console.print("  [dim]Mem clock: unlocked[/]")
        env["mem_clock_locked_mhz"] = "unlocked"

    # Update environment JSON with final locked values
    write_result("00_environment.json", env)
    console.print(f"\n[bold green]Clocks locked.[/]")


if __name__ == "__main__":
    main()
