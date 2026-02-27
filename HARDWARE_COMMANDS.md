# CU Benchmark Suite — Hardware Commands Reference

Quick reference for manually inspecting and locking GPU state.
Use these to understand your hardware before running the automated suite.

---

## NVIDIA — Discover What To Lock

```bash
# GPU model + count
nvidia-smi -L

# Base GPU clock (this is what we lock to)
nvidia-smi --query-gpu=clocks.default_applications.graphics --format=csv,noheader,nounits

# Base memory clock (this is what we lock to)
nvidia-smi --query-gpu=clocks.default_applications.memory --format=csv,noheader,nounits

# Default power limit / TDP (this is what we lock to)
nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits

# Max power limit (absolute ceiling)
nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits

# Compute capability (9.0 = Hopper, 8.0 = Ampere, 8.9 = Ada)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# ECC state
nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader

# All supported clock frequencies
nvidia-smi -q -d SUPPORTED_CLOCKS

# Full hardware dump
nvidia-smi -q
```

## NVIDIA — Lock Everything

```bash
# 1. Persistence mode (keeps driver loaded between calls)
nvidia-smi -pm 1

# 2. Lock GPU clock to base frequency
#    Replace 1410 with YOUR base clock from the discover step
nvidia-smi -lgc 1410,1410

# 3. Lock memory clock
#    Replace 1593 with YOUR base mem clock from the discover step
#    IMPORTANT: On Hopper (H100/H200, compute_cap 9.x), use deferred mode:
nvidia-smi -lmc 1593,1593                              # Ampere, Ada, etc.
nvidia-smi --lock-memory-clocks-deferred 1593,1593      # Hopper ONLY

# 4. Set power limit to TDP
#    Replace 700 with YOUR default power limit
nvidia-smi -pl 700
```

## NVIDIA — Monitor

```bash
# Temperature (single reading)
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# Temperature + power + clocks (continuous, every 1 second)
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.gr,clocks.mem --format=csv -l 1

# Memory in use (check for other processes)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Running processes on GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# NVLink / interconnect topology
nvidia-smi topo -m

# NVLink status
nvidia-smi nvlink -s
```

## NVIDIA — Unlock / Reset

```bash
nvidia-smi -rgc           # reset GPU clocks
nvidia-smi -rmc           # reset memory clocks
nvidia-smi -pl <default>  # reset power (use value from discover step)
```

---

## AMD — Discover What To Lock

```bash
# GPU model
rocm-smi --showproductname
# or (newer tool):
amd-smi static --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['asic']['market_name'])"

# GPU count
rocm-smi --showcount

# Current + supported clock levels
rocm-smi --showclk sclk
rocm-smi --showclk mclk

# Max power cap
rocm-smi --showmaxpower
# or:
amd-smi static -p --json

# Driver version
rocm-smi --showdriverversion

# ECC / RAS status
rocm-smi --showras
# or:
amd-smi metric -e

# Full dump
rocm-smi -a
```

## AMD — Lock Everything

```bash
# 1. Set performance level to high (stabilizes clocks)
rocm-smi --setperflevel high

# 2. Performance determinism mode (prevents PCC downclocking)
#    MI300X max SCLK = 2100 MHz, set to ~1900 for deterministic behavior
rocm-smi --setperfdeterminism 1900

# 3. Set power limit
#    Replace 750 with YOUR power cap from discover step
amd-smi set -o 750
# or (legacy):
rocm-smi --setpoweroverdrive 750
```

## AMD — Monitor

```bash
# Temperature (single reading)
rocm-smi --showtemp
# or:
amd-smi metric -t

# Continuous temperature monitoring
amd-smi monitor -t

# Temperature + power + clocks (JSON for scripting)
amd-smi metric -t -p -c --json

# Running processes
rocm-smi --showpids
# or:
amd-smi process

# xGMI / Infinity Fabric topology
rocm-smi --showtopo

# xGMI link metrics
amd-smi xgmi --metric
```

## AMD — Unlock / Reset

```bash
rocm-smi -r               # resets ALL: clocks, perf level, power overdrive
```

---

## Side-by-Side Quick Reference

| Action | NVIDIA | AMD |
|--------|--------|-----|
| **Base GPU clock** | `--query-gpu=clocks.default_applications.graphics` | `--showclk sclk` |
| **Base mem clock** | `--query-gpu=clocks.default_applications.memory` | `--showclk mclk` |
| **TDP / power cap** | `--query-gpu=power.default_limit` | `--showmaxpower` |
| **Lock GPU clock** | `-lgc <freq>,<freq>` | `--setperfdeterminism <freq>` |
| **Lock mem clock** | `-lmc <freq>,<freq>` (or `--lock-memory-clocks-deferred` for Hopper) | `--setperflevel high` |
| **Set power limit** | `-pl <watts>` | `amd-smi set -o <watts>` |
| **Read temperature** | `--query-gpu=temperature.gpu` | `--showtemp` or `amd-smi metric -t` |
| **Check ECC** | `--query-gpu=ecc.mode.current` | `--showras` or `amd-smi metric -e` |
| **Show topology** | `topo -m` | `--showtopo` |
| **Check processes** | `--query-compute-apps=pid` | `--showpids` or `amd-smi process` |
| **Reset everything** | `-rgc && -rmc && -pl <default>` | `-r` (single command) |

---

## Notes

- Clock locking requires root or admin privileges on most systems
- Inside Docker with `--gpus all`, locking usually works without extra permissions
- On consumer GPUs, some locking features may not be available
- Always unlock/reset after benchmarking to return GPU to normal operation
- AMD is migrating from `rocm-smi` to `amd-smi` — the suite checks for both
