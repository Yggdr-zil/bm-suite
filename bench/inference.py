#!/usr/bin/env python3
"""
CU Benchmark Suite — Inference Throughput Benchmark
Measures Mtok/hr using vLLM with a standard model configuration.

This benchmark is OPTIONAL — it requires:
  1. vLLM installed
  2. Model weights available at CU_BENCH_MODEL_DIR

If either is missing, the benchmark skips gracefully.
"""
import json
import os
import sys
import time

from _common import RESULTS_DIR, write_result, console, print_header

MODEL_DIR = os.environ.get("CU_BENCH_MODEL_DIR", "/models")
MODEL_NAME = os.environ.get("CU_BENCH_MODEL", "")
BATCH_SIZES = [1, 8, 32]
INPUT_LEN = int(os.environ.get("CU_BENCH_INPUT_LEN", "512"))
OUTPUT_LEN = int(os.environ.get("CU_BENCH_OUTPUT_LEN", "128"))
NUM_PROMPTS_PER_BATCH = int(os.environ.get("CU_BENCH_NUM_PROMPTS", "50"))
WARMUP_PROMPTS = 5


def find_model():
    """Auto-detect model from MODEL_DIR or MODEL_NAME."""
    if MODEL_NAME:
        if os.path.isdir(MODEL_NAME):
            return MODEL_NAME
        full_path = os.path.join(MODEL_DIR, MODEL_NAME)
        if os.path.isdir(full_path):
            return full_path
        return MODEL_NAME

    if os.path.isdir(MODEL_DIR):
        for entry in sorted(os.listdir(MODEL_DIR)):
            candidate = os.path.join(MODEL_DIR, entry)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                return candidate

    return None


def main():
    # Check vLLM
    try:
        from vllm import LLM, SamplingParams
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    except ImportError:
        console.print("[yellow]vLLM not installed. Skipping inference benchmark.[/]")
        console.print("[dim]Install with: pip install vllm[/]")
        write_result("05_inference.json", {"skipped": True, "reason": "vllm_not_installed"})
        return

    import torch
    if not torch.cuda.is_available():
        console.print("[red]No CUDA device. Skipping inference benchmark.[/]")
        write_result("05_inference.json", {"skipped": True, "reason": "no_cuda"})
        return

    model_path = find_model()
    if not model_path:
        console.print(f"[yellow]No model found in {MODEL_DIR} and CU_BENCH_MODEL not set.[/]")
        console.print("[dim]Set CU_BENCH_MODEL=meta-llama/Llama-2-70b-hf (or path to local model)[/]")
        write_result("05_inference.json", {"skipped": True, "reason": "no_model", "searched": MODEL_DIR})
        return

    gpu_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print_header(
        f"Inference — {device_name} x{gpu_count}",
        f"Model: {model_path}\nInput: {INPUT_LEN} tok | Output: {OUTPUT_LEN} tok"
    )

    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=gpu_count,
            dtype="auto",           # respect model's native dtype / quantization
            max_model_len=INPUT_LEN + OUTPUT_LEN,
            trust_remote_code=True,
        )
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/]")
        write_result("05_inference.json", {"skipped": True, "reason": f"model_load_failed: {e}"})
        return

    sampling_params = SamplingParams(temperature=0, max_tokens=OUTPUT_LEN)

    torch.manual_seed(42)
    max_prompts = max(BATCH_SIZES) * NUM_PROMPTS_PER_BATCH + WARMUP_PROMPTS
    dummy_ids = torch.randint(100, 30000, (max_prompts, INPUT_LEN))
    all_prompts = [{"prompt_token_ids": ids.tolist()} for ids in dummy_ids]

    console.print(f"  [dim]Warmup ({WARMUP_PROMPTS} prompts)...[/]")
    _ = llm.generate(all_prompts[:WARMUP_PROMPTS], sampling_params)

    results = {"model": model_path, "device": device_name, "gpu_count": gpu_count, "batches": {}}

    prompt_offset = WARMUP_PROMPTS
    for batch_size in BATCH_SIZES:
        num_prompts = NUM_PROMPTS_PER_BATCH
        prompts = all_prompts[prompt_offset:prompt_offset + num_prompts]
        prompt_offset += num_prompts

        if len(prompts) < num_prompts:
            console.print(f"  [yellow]Batch {batch_size}: not enough prompts, skipping[/]")
            continue

        console.print(f"  [cyan]Batch {batch_size}[/] ({num_prompts} prompts)...", end=" ")

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_in_tokens = num_prompts * INPUT_LEN
        tps = total_out_tokens / elapsed
        mtok_hr = tps * 3600 / 1_000_000

        batch_result = {
            "num_prompts": num_prompts,
            "total_input_tokens": total_in_tokens,
            "total_output_tokens": total_out_tokens,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(tps, 1),
            "mtok_per_hour": round(mtok_hr, 2),
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
        }
        results["batches"][f"batch_{batch_size}"] = batch_result
        console.print(f"[green]{tps:.1f} tok/s[/] | [green]{mtok_hr:.2f} Mtok/hr[/]")

    # Compute efficiency_factor — the frozen constant for the virtual cluster bridge
    efficiency_cal = _compute_efficiency(model_path, results)
    if efficiency_cal:
        results["efficiency_calibration"] = efficiency_cal
        ef = efficiency_cal.get("efficiency_factor")
        if ef:
            console.print(f"\n  [bold]Efficiency factor:[/] [green]{ef:.4f}[/]  "
                          f"[dim](roofline: {efficiency_cal.get('roofline_single_tok_s')} tok/s, "
                          f"model: {efficiency_cal.get('weight_bytes_gb')} GB, "
                          f"{efficiency_cal.get('params_estimated_B')}B params)[/]")

    write_result("05_inference.json", results)


def _compute_efficiency(model_path, results):
    """Compute roofline efficiency factor from measured throughput vs. HBM bandwidth.

    efficiency_factor = (measured tok/s per sequence) / (membw_bps / weight_bytes)

    In the bandwidth-bound regime this is approximately constant across batch sizes.
    This is the frozen constant used in the virtual cluster bridge (patent Claim 24):
        virtual_baseline = (ref_bw / weight_bytes) × efficiency_factor × concurrency
    """
    import json as _json

    # ── 1. Measured HBM bandwidth from membw benchmark ──
    membw_path = os.path.join(RESULTS_DIR, "02_membw.json")
    membw_gbps = None
    if os.path.exists(membw_path):
        try:
            with open(membw_path) as f:
                mbw = _json.load(f)
            membw_gbps = (mbw.get("clone_large") or {}).get("gbps") or \
                         (mbw.get("clone_primary") or {}).get("gbps")
        except Exception:
            pass

    if not membw_gbps:
        return {"skipped": True, "reason": "membw results not available — run benchmarks in order"}

    # ── 2. Model weight size from config.json ──
    config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if not config_path or not os.path.exists(config_path):
        return {"skipped": True, "reason": "model config.json not found"}

    try:
        with open(config_path) as f:
            cfg = _json.load(f)

        h     = cfg.get("hidden_size", 0)
        L     = cfg.get("num_hidden_layers", 0)
        I     = cfg.get("intermediate_size", 0)
        V     = cfg.get("vocab_size", 0)
        n_kv  = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads", 0)
        n_q   = cfg.get("num_attention_heads", 0)
        d     = h // n_q if n_q else 128

        # Standard transformer param count (Q + K + V + O + gate/up/down FFN + embedding)
        attn   = (h * h + h * n_kv * d + h * n_kv * d + h * h) * L
        ffn    = 3 * h * I * L  # SwiGLU: gate + up + down
        emb    = V * h
        params = attn + ffn + emb

        # Detect quantization dtype for correct weight_bytes calculation.
        # Methodology §6.8 calibrates efficiency_factor using INT4 (0.5 bytes/param).
        # For pre-quantized models (AWQ/GPTQ/bitsandbytes INT4): bytes_per_param = 0.5.
        # For FP16/BF16: bytes_per_param = 2.0.
        bytes_per_param = 2.0
        weight_dtype = "fp16"
        try:
            quant_cfg = cfg.get("quantization_config") or {}
            bits = (quant_cfg.get("bits") or quant_cfg.get("num_bits") or
                    quant_cfg.get("w_bit") or quant_cfg.get("quant_type", ""))
            if bits in (4, "int4", "nf4", "fp4") or str(bits).lower() in ("q4_0", "q4_k"):
                bytes_per_param = 0.5
                weight_dtype = "int4"
            elif bits in (8, "int8", "fp8"):
                bytes_per_param = 1.0
                weight_dtype = "int8"
            else:
                torch_dtype = cfg.get("torch_dtype", "float16")
                weight_dtype = str(torch_dtype).replace("torch.", "")
        except Exception:
            pass

        weight_bytes = params * bytes_per_param
        weight_gb    = weight_bytes / 1e9
    except Exception as e:
        return {"skipped": True, "reason": f"config.json parse failed: {e}"}

    # ── 3. Efficiency at each measured batch size ──
    # efficiency = (tok/s per sequence) / roofline_single
    # Should be approximately constant across batch sizes in the bandwidth-bound regime.
    # Variation indicates compute-bound batches (typically small models at large batch).
    membw_bps       = membw_gbps * 1e9
    roofline_single = membw_bps / weight_bytes  # theoretical tok/s for 1 sequence

    per_batch = {}
    best_efficiency = None  # computed at largest (most saturated) batch first

    for b in [32, 8, 1]:
        key = f"batch_{b}"
        r   = (results.get("batches") or {}).get(key)
        if not r:
            continue
        tps_per_seq = r["tokens_per_second"] / b
        eff = round(tps_per_seq / roofline_single, 4)
        per_batch[key] = eff
        if best_efficiency is None:
            best_efficiency = eff  # use largest batch as canonical value

    if best_efficiency is None:
        return {"skipped": True, "reason": "no batch inference results found"}

    return {
        "efficiency_factor":       round(best_efficiency, 4),
        "efficiency_per_batch":    per_batch,
        "roofline_single_tok_s":   round(roofline_single, 1),
        "membw_gbps":              membw_gbps,
        "weight_bytes_gb":         round(weight_gb, 2),
        "weight_dtype":            weight_dtype,
        "bytes_per_param":         bytes_per_param,
        "params_estimated_B":      round(params / 1e9, 2),
        "note": (
            "efficiency_factor = (measured tok/s / batch_size) / (membw_bps / weight_bytes). "
            "Constant across batch sizes when bandwidth-bound. "
            "Frozen constant for virtual cluster bridge: "
            "virtual_baseline = (ref_bw / weight_bytes) * efficiency_factor * concurrency."
        ),
    }


if __name__ == "__main__":
    main()
