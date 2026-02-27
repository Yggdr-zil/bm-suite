# CU Benchmark Suite v1.0

Standardized GPU/accelerator benchmark suite for the Compute Unit (CU) Index.
Physically measures performance specs from hardware — does not trust published numbers.

Supports NVIDIA (cuda/nvidia-smi) and AMD (rocm/rocm-smi) with auto-detection.

## What It Measures

The suite extracts **5 performance dimensions** from any GPU/accelerator:

| # | Metric | Method | Output |
|---|--------|--------|--------|
| 1 | **FP Throughput** | GEMM (matrix multiply) at FP32, FP16, BF16, FP8 | TFLOPS per precision |
| 2 | **Memory Bandwidth** | Tensor clone + element-wise multiply | GB/s |
| 3 | **VRAM Capacity** | Binary search allocation with context accounting | GB (verified working) |
| 4 | **Interconnect BW** | NCCL/RCCL all_reduce across GPUs | GB/s bus bandwidth |
| 5 | **Inference Throughput** | vLLM with standard model config | Mtok/hr |

Benchmarks 1-4 are micro-benchmarks (measure raw hardware specs).
Benchmark 5 is a macro-benchmark (measures real-world throughput).
Both feed into the CU scoring formulas.

## How It Works

### High-Level Flow

```
[Preflight] → [Telemetry] → [Warmup] → [Benchmarks 1-5] → [Unlock] → [Report]
```

1. **Preflight** (`preflight.sh`) — Auto-detects platform (NVIDIA/AMD), discovers hardware
   specs dynamically, locks GPU clocks + memory clocks + power limit for deterministic
   results, checks for clean GPU, writes `00_environment.json`.

2. **Telemetry** (`telemetry.sh`) — Background process logging temp/power/clocks every
   second to CSV. Runs for the entire benchmark duration. Proves thermal and power
   conditions during the test.

3. **Thermal Warmup** (`thermal_gate.py`) — Runs sustained GEMM until the GPU reaches
   thermal steady state (junction temp delta < 2C over 30 seconds). Cold GPUs boost
   higher than sustained — this eliminates that variable.

4. **GEMM** (`gemm.py`) — Square matrix multiply (default 8192x8192) at each precision.
   20 warmup iterations (primes cuBLAS auto-tuner), 10,000 measured iterations.
   TFLOPS = 2 * M * N * K / time / 1e12. Tests tensor cores (FP16/BF16/FP8) and
   CUDA cores (FP32) separately. Stores timing percentiles + sampled drift series.

5. **Memory Bandwidth** (`membw.py`) — Tensor clone (read+write = 2x tensor size) and
   element-wise multiply (read A + read B + write C = 3x). Tests at 10% and 25% of
   VRAM to ensure we exceed L2 cache and hit main memory. 5,000 iterations with
   trimmed-mean aggregation and sampled bandwidth drift series.

6. **VRAM** (`vram.py`) — Binary search allocation (30 iterations) finds the true
   allocatable ceiling. Adds CUDA context overhead back in — the runtime's memory is
   implicitly verified by the process being alive. Result = all working bytes on the chip.

7. **Interconnect** (`interconnect.sh`) — NCCL (NVIDIA) or RCCL (AMD) `all_reduce_perf`
   sweep from 1MB to 2GB. Measures bus bandwidth at saturation. Skips on single GPU.

8. **Inference** (`inference.py`) — vLLM with deterministic prompts (fixed seed),
   greedy decoding, measures tokens/sec at batch sizes 1/8/32. Optional — skips
   gracefully if vLLM or model weights aren't available.

9. **Report** (`report.py`) — Merges all JSON results into `benchmark_report.json`
   with a SHA256 integrity hash. Loads `telemetry.csv` + `stage_events.csv` to
   compute per-stage thermal profiles and detect throttle events.

10. **Plot** (`plot.py`) — Auto-generates PNG charts and `plots/summary.html` from
    the report: telemetry overview (temp/power/clock with stage boundaries), GEMM TFLOPS
    bars, timing drift series per precision, membw bars + drift, VRAM breakdown,
    interconnect sweep curve.

### Why Lock the Hardware

GPUs dynamically boost clocks based on temperature, power headroom, and workload.
A cold H100 can briefly hit 2.1 GHz then throttle to 1.6 GHz. Locking clocks
to the sustained base frequency ensures:
- Run 1 and run 100 produce the same number
- GPU A and GPU B are compared at the same operating point
- The measurement reflects sustained capability, not transient boost

### Why Warmup Matters

A GPU at 35C boosts differently than one at 75C. The thermal gate runs sustained
GEMM until the junction temperature stabilizes (delta < 2C over 30 seconds).
After warmup, the GPU is at its steady-state thermal operating point — the benchmarks
that follow reflect sustained, not peak, performance.

### VRAM: Why We Don't Trust Reported Numbers

`nvidia-smi` and `torch.cuda.get_device_properties()` report what the driver claims.
This can be wrong — bad memory banks mapped out at the factory, ECC reservations,
firmware bugs, Samsung vs Micron HBM3 on the same SKU.

We physically allocate with `torch.empty()` in a binary search to find the true
ceiling, then add back the CUDA context overhead (which is implicitly verified —
if those pages were bad, the process would have crashed):

```
verified_working_vram = max_allocatable + cuda_context_overhead
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch with CUDA support
- `rich` (terminal UI)
- `nvidia-smi` or `rocm-smi`
- (Optional) vLLM + model weights for inference benchmark
- (Optional) nccl-tests compiled for interconnect benchmark

### Setup

```bash
cd bm-suite
uv venv && source .venv/bin/activate
uv pip install torch rich
```

### Run Everything (Full Orchestrated Suite)

```bash
bash run_local.sh
```

This runs the complete pipeline: preflight -> telemetry -> warmup -> all 5 benchmarks
-> unlock -> report. Results land in `results/run_{timestamp}/`.

### Run Individual Benchmarks

Every script is self-contained. Run from the `bench/` directory:

```bash
cd bench

# Each creates its own timestamped results dir:
python3 gemm.py                    # → results/run_XXXXX/01_gemm.json
python3 membw.py                   # → results/run_XXXXX/02_membw.json
python3 vram.py                    # → results/run_XXXXX/03_vram.json

# Or point them all at the same dir:
export RESULTS_DIR=~/my_run
python3 gemm.py
python3 membw.py
python3 vram.py
python3 report.py                  # merges everything into benchmark_report.json
```

### Find Sustained Clock (Blackwell / Consumer GPUs)

If `Default Applications Clocks` is deprecated on your GPU (RTX 50-series),
run this first to find your sustained clock frequency:

```bash
python3 bench/find_sustained_clock.py
# Outputs: "Lock recommendation: sudo nvidia-smi -lgc {freq},{freq}"
```

### Docker (For Cloud Deployment)

```bash
docker build -t cu-bench .
docker run --gpus all --cap-add=SYS_ADMIN \
  --ipc=host --ulimit memlock=-1:-1 \
  -v /path/to/models:/models \
  -v /path/to/results:/results \
  cu-bench

# Quick mode (faster, fewer iterations — for validation):
docker run --gpus all --cap-add=SYS_ADMIN \
  --ipc=host --ulimit memlock=-1:-1 \
  -e CU_QUICK=1 \
  -v /path/to/results:/results \
  cu-bench
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RESULTS_DIR` | `./results/run_{timestamp}` | Output directory for benchmark results |
| `CU_QUICK` | (unset) | Set to `1` for fast mode (1000 GEMM / 500 membw iters) |
| `CU_VRAM_BUDGET` | `0.90` | Max fraction of free VRAM to use (safety cap) |
| `CU_GEMM_DIM` | `8192` | GEMM matrix dimension (use 4096 for small GPUs) |
| `CU_GEMM_ITERS` | `10000` | GEMM measured iterations (CU_QUICK → 1000) |
| `CU_GEMM_WARMUP` | `20` | GEMM warmup iterations |
| `CU_GEMM_TRIM_PCT` | `5` | % of slowest samples to trim from GEMM mean |
| `CU_MEMBW_ITERS` | `5000` | Memory bandwidth measured iterations (CU_QUICK → 500) |
| `CU_MEMBW_WARMUP` | `10` | membw warmup iterations |
| `CU_MEMBW_TRIM_PCT` | `5` | % of slowest samples to trim from membw mean |
| `CU_WARMUP_MIN_SECS` | `300` | Min thermal soak time (5 min mandatory floor) |
| `CU_WARMUP_MAX_SECS` | `600` | Max thermal warmup time |
| `CU_GPU_CLOCK` | (none) | Override sustained clock detection, lock to this MHz |
| `CU_MEM_CLOCK` | (none) | Override memory clock lock MHz |
| `CU_BENCH_MODEL` | (none) | Model name/path for inference benchmark |
| `CU_BENCH_MODEL_DIR` | `/models` | Directory containing model weights |
| `CU_BENCH_INPUT_LEN` | `512` | Inference benchmark input token length |
| `CU_BENCH_OUTPUT_LEN` | `128` | Inference benchmark output token length |

## Output Format

Every JSON result file includes a `_meta` block for provenance:

```json
{
  "_meta": {
    "run_id": "20260226T171518Z_641c9b45",
    "timestamp": "2026-02-26T17:15:18+00:00",
    "hostname": "dgx-01",
    "pid": 12345,
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "gpu_count": 8,
    "pytorch_version": "2.6.0",
    "cuda_version": "12.4"
  },
  ...benchmark-specific data...
}
```

The final `benchmark_report.json` merges all benchmarks and includes:
- `measured_specs` — flat dict of primary measurements for the CU formula
- `benchmarks_completed` — which benchmarks ran successfully
- `telemetry` — loaded CSV rows + per-stage thermal profiles + throttle events
- `integrity_sha256` — SHA256 of the report (covers telemetry + all measurements)
- `detailed` — full data from each benchmark including timing percentiles + drift series

## File Reference

```
bm-suite/
├── README.md                  ← you are here
├── Dockerfile                 ← CUDA 12.4 container with PyTorch + vLLM + nccl-tests
├── run_local.sh               ← local runner (no Docker), calls bench/run_all.sh
├── HARDWARE_COMMANDS.md       ← NVIDIA vs AMD command reference (discover/lock/monitor/unlock)
├── bench/
│   ├── _common.py             ← shared runtime: RUN_ID, RESULTS_DIR, write_result(), VRAM budget
│   ├── preflight.sh           ← [0] detect platform, discover specs, lock clocks/power
│   ├── telemetry.sh           ← [1] background temp/power/clock CSV logger (1 Hz)
│   ├── thermal_gate.py        ← [2] sustained GEMM until thermal steady state
│   ├── gemm.py                ← [3] FP32/FP16/BF16/FP8 throughput via matrix multiply
│   ├── membw.py               ← [4] memory bandwidth via clone + element-wise multiply
│   ├── vram.py                ← [5] binary search allocation with context accounting
│   ├── interconnect.sh        ← [6] NCCL/RCCL all_reduce bus bandwidth
│   ├── inference.py           ← [7] vLLM Mtok/hr (optional)
│   ├── report.py              ← merges all results, loads telemetry, integrity seal
│   ├── plot.py                ← auto-generates PNG charts + summary.html
│   ├── run_all.sh             ← orchestrator: runs [0]-[9] in sequence
│   └── find_sustained_clock.py ← utility: find steady-state clock for Blackwell/consumer GPUs
└── results/                   ← timestamped run directories (gitignored)
    └── run_{timestamp}/
        ├── benchmark_report.json  ← primary output, SHA256-sealed
        ├── telemetry.csv          ← 1Hz GPU temp/power/clock throughout run
        ├── stage_events.csv       ← stage start/end timestamps for telemetry alignment
        └── plots/
            ├── summary.html       ← all charts in one page
            ├── telemetry_overview.png
            ├── gemm_tflops.png
            ├── gemm_drift_{fp32,fp16,bf16,fp8}.png
            ├── membw.png
            ├── vram.png
            └── interconnect.png
```

## Running Alongside Other GPU Workloads

The suite respects a VRAM budget (`CU_VRAM_BUDGET`, default 90% of free VRAM).
If you're running training or other workloads:

```bash
export CU_VRAM_BUDGET=0.5    # only use 50% of free VRAM
python3 bench/gemm.py         # skips if allocation exceeds budget
```

- **GEMM/membw**: check budget before allocating matrices, skip gracefully if too tight
- **VRAM probe**: searches within free memory only, never evicts other allocations
- **Interconnect/inference**: use their own VRAM but don't pre-check budget — run these on clean GPUs

For accurate results, always benchmark on a **clean GPU** with no other processes running.
The suite warns you if the GPU isn't clean but continues anyway (results are flagged).

## Clock Locking Notes

| GPU Family | GPU Clock | Memory Clock | Power Limit | Notes |
|------------|-----------|--------------|-------------|-------|
| Datacenter (A100, H100) | `nvidia-smi -lgc` | `-lmc` or `--lock-memory-clocks-deferred` (Hopper) | `-pl` | Full support, no sudo |
| Consumer (RTX 30/40/50) | `sudo nvidia-smi -lgc` | `sudo nvidia-smi -lmc` | `sudo nvidia-smi -pl` | Requires root |
| Blackwell consumer (RTX 50) | Same as above | Same | Same | `Default Applications Clocks` deprecated — use `find_sustained_clock.py` |
| AMD (MI300X/MI325X) | `rocm-smi --setperfdeterminism` | N/A (HBM is fixed) | `rocm-smi --setpoweroverdrive` | |

## Unlocking After Benchmarks

`run_all.sh` automatically unlocks hardware when done. If you ran scripts manually:

```bash
# NVIDIA
sudo nvidia-smi -rgc           # reset GPU clocks
sudo nvidia-smi -rmc           # reset memory clocks

# AMD
sudo rocm-smi -r               # reset all
```
