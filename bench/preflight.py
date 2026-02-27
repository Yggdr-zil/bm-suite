#!/usr/bin/env python3
"""
CU Benchmark Suite — Preflight: Detect + Discover
Discovers hardware parameters, sets fans, writes initial environment.
Does NOT lock clocks — that happens AFTER thermal warmup + sustained clock detection.
Writes 00_environment.json and _platform.sh (for shell scripts to source).
"""
import json
import os
import platform as plat
import subprocess
import sys

from rich.table import Table

from _common import RESULTS_DIR, console, print_header, write_result


def run(cmd, fallback=""):
    """Run a command, return stdout stripped. Returns fallback on failure."""
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        return out if out else fallback
    except Exception:
        return fallback


def nvidia_query(field, nounits=True):
    """Query nvidia-smi for a single GPU field."""
    cmd = ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"]
    if nounits:
        cmd[-1] += ",nounits"
    out = run(cmd)
    return out.split("\n")[0].strip() if out else ""


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


def parse_nvlink_topology(raw):
    """Parse nvidia-smi topo -m matrix into structured connections list.

    Returns list of {from_gpu, to_gpu, link_type, nvlink_count} dicts,
    or {parsed: False, raw: raw} if the matrix cannot be parsed.

    Link type codes from nvidia-smi: NV1..NV18 = NVLink gen/count,
    PHB = PCIe-Host Bridge, NODE = NUMA node, SYS = cross-socket.
    """
    try:
        lines = [l for l in raw.splitlines() if l.strip()]
        # Find header row — starts with GPU\t or contains "GPU0"
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("GPU") and "\t" in line:
                header_idx = i
                break
        if header_idx is None:
            return {"parsed": False, "raw": raw}

        # Parse header GPU indices: "GPU0\tGPU1\t..."
        header_parts = lines[header_idx].strip().split("\t")
        col_gpus = []
        for p in header_parts:
            p = p.strip()
            if p.startswith("GPU"):
                try:
                    col_gpus.append(int(p[3:]))
                except ValueError:
                    pass

        connections = []
        for line in lines[header_idx + 1:]:
            parts = line.strip().split("\t")
            if not parts or not parts[0].startswith("GPU"):
                continue
            try:
                from_gpu = int(parts[0][3:].split()[0])
            except (ValueError, IndexError):
                continue
            for col_idx, link in enumerate(parts[1:len(col_gpus) + 1]):
                link = link.strip()
                if col_idx >= len(col_gpus):
                    break
                to_gpu = col_gpus[col_idx]
                if from_gpu == to_gpu or link in ("", "X", "-"):
                    continue
                # Count NVLink lanes (NV4 → 4 lanes, NV12 → 12, etc.)
                nvlink_count = None
                if link.startswith("NV"):
                    try:
                        nvlink_count = int(link[2:])
                    except ValueError:
                        pass
                connections.append({
                    "from_gpu": from_gpu,
                    "to_gpu": to_gpu,
                    "link_type": link,
                    "nvlink_count": nvlink_count,
                })
        return {"parsed": True, "connections": connections, "raw": raw}
    except Exception as e:
        return {"parsed": False, "error": str(e), "raw": raw}


def detect_platform():
    """Detect GPU platform."""
    if run(["nvidia-smi", "-L"]):
        return "nvidia"
    if run(["rocm-smi", "--showcount"]):
        return "amd"
    console.print("[bold red]FATAL: No GPU detected (need nvidia-smi or rocm-smi)[/]")
    sys.exit(1)


def discover_nvidia():
    """NVIDIA: discover hardware, set fans. No clock locking."""
    env = {"platform": "nvidia"}

    # ─── Discover ───
    env["gpu_model"] = nvidia_query("name", nounits=False)
    env["gpu_count"] = int(nvidia_query("count") or "1")
    env["driver_version"] = nvidia_query("driver_version", nounits=False)
    env["compute_capability"] = nvidia_query("compute_cap", nounits=False)
    env["ecc_mode"] = nvidia_query("ecc.mode.current", nounits=False) or "N/A"

    # Clocks (discovery only — locking happens later)
    default_gpu_clk = nvidia_query("clocks.default_applications.graphics")
    default_mem_clk = nvidia_query("clocks.default_applications.memory")
    max_gpu_clk = nvidia_query("clocks.max.graphics")
    max_mem_clk = nvidia_query("clocks.max.memory")

    env["gpu_clock_max_mhz"] = int(max_gpu_clk) if max_gpu_clk else 0
    env["mem_clock_max_mhz"] = int(max_mem_clk) if max_mem_clk else 0

    # Check if default applications clocks are deprecated
    clocks_deprecated = False
    if "deprecated" in default_gpu_clk.lower() or "error" in default_gpu_clk.lower() or not default_gpu_clk:
        clocks_deprecated = True
    env["clocks_deprecated"] = clocks_deprecated

    # Store default clocks for lock step (if not deprecated)
    user_gpu_clk = os.environ.get("CU_GPU_CLOCK", "").strip()
    user_mem_clk = os.environ.get("CU_MEM_CLOCK", "").strip()

    if user_gpu_clk:
        env["target_gpu_clock"] = int(user_gpu_clk)
        env["clock_source"] = "user (CU_GPU_CLOCK)"
    elif not clocks_deprecated and default_gpu_clk:
        env["target_gpu_clock"] = int(default_gpu_clk)
        env["clock_source"] = "default applications clock"
    else:
        env["target_gpu_clock"] = None
        env["clock_source"] = "pending (sustained clock detection after warmup)"

    if user_mem_clk:
        env["target_mem_clock"] = int(user_mem_clk)
    elif not clocks_deprecated and default_mem_clk:
        env["target_mem_clock"] = int(default_mem_clk)
    else:
        env["target_mem_clock"] = None

    # Power limit
    default_pl = nvidia_query("power.default_limit")
    pl_int = int(float(default_pl)) if default_pl else 0
    env["power_limit_watts"] = pl_int

    # Memory / processes check
    used_mem = nvidia_query("memory.used")
    used_mib = int(float(used_mem)) if used_mem else 0
    procs_out = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    proc_count = len([p for p in procs_out.split("\n") if p.strip()]) if procs_out else 0

    # Temperature
    start_temp = int(nvidia_query("temperature.gpu") or "0")
    env["start_temp_c"] = start_temp

    # Serials
    env["gpu_serials"] = nvidia_query("serial", nounits=False) or "N/A"

    # Topology — NVLink mesh and PCIe mapping (patent: Claim 27 full-mesh evidence)
    topo_raw = run(["nvidia-smi", "topo", "-m"]) or "N/A"
    env["nvlink_topology_raw"] = topo_raw
    env["nvlink_topology"] = parse_nvlink_topology(topo_raw)
    env["nvlink_status"] = run(["nvidia-smi", "nvlink", "-s"]) or "N/A"

    # ─── Display header ───
    print_header(
        f"Preflight — {env['gpu_model']}",
        f"x{env['gpu_count']} | Compute {env['compute_capability']} | Driver {env['driver_version']}"
    )

    if used_mib > 500 or proc_count > 0:
        console.print(f"  [yellow]WARNING: GPU not clean ({used_mib} MiB used, {proc_count} processes)[/]")

    if clocks_deprecated:
        console.print(f"  [yellow]Default Applications Clocks deprecated — will auto-detect after warmup[/]")

    # ─── Fans ───
    fan_set = False
    try:
        help_out = subprocess.check_output(
            ["nvidia-smi", "--help"], text=True, stderr=subprocess.STDOUT, timeout=5
        )
        has_fan_cmd = "--fan-speed" in help_out or "--auto-fan-speed" in help_out
    except Exception:
        has_fan_cmd = False

    if has_fan_cmd:
        console.print("\n[bold]Setting fans to max...[/]")
        for fan_idx in range(4):
            if nvidia_lock(["-i", "0", f"--fan-speed={100}", f"--id={fan_idx}"],
                           f"Fan {fan_idx}: 100%"):
                fan_set = True
            else:
                break
        if not fan_set:
            nvidia_lock(["-i", "0", "--fan-speed=100"], "Fans: 100%")
    else:
        console.print("\n[dim]Fan control: EC / BMC managed (no nvidia-smi fan support)[/]")
    env["fans_maxed"] = fan_set

    # ─── Persistence mode (safe to enable early) ───
    nvidia_lock(["-pm", "1"], "Persistence mode: ON")

    # ─── Power limit (set early so warmup runs at correct TDP) ───
    nvidia_lock(["-pl", str(pl_int)], f"Power limit: {pl_int} W")

    # ─── Summary ───
    info_table = Table(show_header=False, show_edge=False, padding=(0, 2))
    info_table.add_column(style="bold", width=16)
    info_table.add_column()

    gpu_clk_display = env.get("target_gpu_clock")
    clk_src = env["clock_source"]
    info_table.add_row("GPU Clock", f"[green]{gpu_clk_display}[/] MHz  [dim]({clk_src})[/]" if gpu_clk_display else f"[yellow]pending[/]  [dim]({clk_src})[/]")
    info_table.add_row("Mem Clock", f"[green]{env.get('target_mem_clock')}[/] MHz" if env.get("target_mem_clock") else "[dim]unlocked[/]")
    info_table.add_row("Max Clocks", f"[dim]{max_gpu_clk} / {max_mem_clk} MHz[/]")
    info_table.add_row("TDP", f"{pl_int} W")
    info_table.add_row("ECC", env["ecc_mode"])
    info_table.add_row("Fans", "[green]100%[/]" if fan_set else "[dim]auto (EC/BMC)[/]")
    info_table.add_row("Start Temp", f"{start_temp}C")
    info_table.add_row("VRAM Used", f"{used_mib} MiB  ({proc_count} compute processes)")
    console.print(info_table)

    # Clocks NOT locked — that happens after warmup + sustained clock detection
    env["gpu_clock_locked_mhz"] = "pending"
    env["mem_clock_locked_mhz"] = "pending"

    return env


def discover_amd():
    """AMD: discover hardware. Clock locking happens here since AMD doesn't need sustained clock detection."""
    env = {"platform": "amd"}
    has_amdsmi = bool(run(["amd-smi", "version"]))

    if has_amdsmi:
        try:
            static = json.loads(run(["amd-smi", "static", "--json"]))
            env["gpu_model"] = static[0].get("asic", {}).get("market_name", "unknown")
        except Exception:
            env["gpu_model"] = "unknown"
    else:
        env["gpu_model"] = run(["rocm-smi", "--showproductname"]) or "unknown"

    try:
        import torch
        env["gpu_count"] = torch.cuda.device_count()
    except Exception:
        env["gpu_count"] = 1

    env["driver_version"] = run(["rocm-smi", "--showdriverversion"]) or "unknown"
    env["compute_capability"] = "N/A"
    env["ecc_mode"] = "N/A"
    env["clocks_deprecated"] = False

    print_header(f"Preflight — {env['gpu_model']}", f"x{env['gpu_count']} | Driver {env['driver_version']}")

    # AMD: lock immediately (determinism mode doesn't need sustained clock detection)
    console.print("\n[bold]Locking hardware...[/]")
    try:
        subprocess.check_call(["rocm-smi", "--setperflevel", "high"], stderr=subprocess.DEVNULL)
        console.print("  [green]Perf level: high[/]")
    except Exception:
        console.print("  [yellow]Could not set perf level[/]")

    try:
        subprocess.check_call(["rocm-smi", "--setperfdeterminism", "1900"], stderr=subprocess.DEVNULL)
        console.print("  [green]Determinism: 1900 MHz[/]")
    except Exception:
        console.print("  [yellow]Could not set determinism[/]")

    env["gpu_clock_locked_mhz"] = 1900
    env["gpu_clock_max_mhz"] = 1900
    env["mem_clock_locked_mhz"] = "N/A"
    env["mem_clock_max_mhz"] = "N/A"
    env["power_limit_watts"] = 750
    env["start_temp_c"] = 0
    env["gpu_serials"] = "N/A"

    # Topology — xGMI link bandwidth + supported clock levels (methodology §HMI query)
    env["gpu_clocks_sclk"] = run(["rocm-smi", "--showclk", "sclk"]) or "N/A"
    if has_amdsmi:
        env["xgmi_metric"] = run(["amd-smi", "xgmi", "--metric"]) or "N/A"
    else:
        env["xgmi_metric"] = "N/A"

    if has_amdsmi:
        try:
            metrics = json.loads(run(["amd-smi", "metric", "-t", "--json"]))
            env["start_temp_c"] = metrics[0].get("temperature", {}).get("hotspot", 0)
        except Exception:
            pass

    return env


def main():
    platform = detect_platform()

    if platform == "nvidia":
        env = discover_nvidia()
    else:
        env = discover_amd()

    # Add host info
    env["host_kernel"] = plat.release()
    env["host_os"] = plat.platform()
    env["python_version"] = plat.python_version()
    env["pytorch_version"] = ""
    env["cuda_runtime"] = ""
    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["cuda_runtime"] = torch.version.cuda or "N/A"
    except ImportError:
        pass

    # Write environment JSON
    write_result("00_environment.json", env)

    # Write _platform.sh for shell scripts to source
    platform_sh = os.path.join(RESULTS_DIR, "_platform.sh")
    with open(platform_sh, "w") as f:
        f.write(f'export PLATFORM="{env["platform"]}"\n')
        f.write(f'export GPU_COUNT="{env["gpu_count"]}"\n')
        f.write(f'export GPU_MODEL="{env["gpu_model"]}"\n')
        f.write(f'export CLOCKS_DEPRECATED="{env.get("clocks_deprecated", False)}"\n')

    console.print(f"\n[bold green]Preflight complete.[/] Start temp: {env.get('start_temp_c', '?')}C")
    console.print(f"[dim]Clocks will be locked after thermal warmup + sustained clock detection.[/]")


if __name__ == "__main__":
    main()
