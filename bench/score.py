#!/usr/bin/env python3
"""CU Benchmark Suite — eCU Scoring Engine

Computes empirical Compute Unit scores (eTCU / eICU / eCU) from benchmark
measurements using v4 physics-constrained exponents against an 8x H100 SXM
cluster reference.

1 eCU = 1 hour of an 8x H100 SXM cluster.

Patent pending — Joseph Januszewski, filed Jan 9, 2026.

Usage:
    python3 score.py                          # scores RESULTS_DIR/benchmark_report.json
    python3 score.py /path/to/report.json     # scores a specific report
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


# ─── Reference Unit: 8x H100 SXM Cluster ────────────────────────
REF = {
    "fp16": 7916.0,       # 989.5 × 8
    "fp8": 15832.0,       # 1979.0 × 8
    "fp32": 536.0,        # 67.0 × 8
    "vram": 640.0,        # 80.0 × 8
    "membw": 26800.0,     # 3350.0 × 8
    "ic_bw": 900.0,       # per-GPU NVLink (NOT multiplied)
}

# ─── v4 Exponents (physics-constrained) ─────────────────────────
# TCU = compute-bound: fp > bw (roofline model)
# ICU = memory-bound: bw >> fp
TCU_EXP = {"fp": 0.523552, "bw": 0.481565, "vram": 0.073083, "ic": 0.127102}
ICU_EXP = {"fp": 0.362984, "bw": 0.462984, "vram": 0.250463, "ic": 0.156754}

# ─── FP Precision Sub-Weights ───────────────────────────────────
FP_W = {"primary": 0.75, "secondary": 0.03, "tertiary": 0.22}


INTEGRITY_FIELDS = {"integrity_sha256", "integrity_method"}


def compute_ecu_scores(measured: dict) -> dict:
    """Compute eTCU, eICU, eCU from measured benchmark specs.

    Args:
        measured: dict with keys fp16_tflops, fp8_tflops, fp32_tflops,
                  vram_usable_gb, membw_gbps, interconnect_bw_gbps.

    Returns:
        dict with eTCU, eICU, eCU (rounded to 6 dp), plus metadata.
    """
    fp16 = measured.get("fp16_tflops") or 0.0
    fp8 = measured.get("fp8_tflops")
    fp32 = measured.get("fp32_tflops") or 0.0
    membw = measured.get("membw_gbps") or 0.0
    vram = measured.get("vram_usable_gb") or 0.0
    ic_bw = measured.get("interconnect_bw_gbps") or 0.0

    # Precision ratios vs reference
    fp16_r = fp16 / REF["fp16"]
    fp32_r = fp32 / REF["fp32"]

    # FP8 fallback: if FP8 is None or 0, FP16 absorbs FP8 weight
    fp8_fallback = fp8 is None or fp8 == 0
    if not fp8_fallback:
        fp8_r = fp8 / REF["fp8"]
        # TCU: FP16 primary
        fp_ratio_tcu = FP_W["primary"] * fp16_r + FP_W["secondary"] * fp8_r + FP_W["tertiary"] * fp32_r
        # ICU: FP8 primary
        fp_ratio_icu = FP_W["primary"] * fp8_r + FP_W["secondary"] * fp16_r + FP_W["tertiary"] * fp32_r
    else:
        # No-FP8 fallback: FP16 absorbs FP8 weight → 0.78 × fp16_r + 0.22 × fp32_r
        fp_ratio_tcu = 0.78 * fp16_r + 0.22 * fp32_r
        fp_ratio_icu = 0.78 * fp16_r + 0.22 * fp32_r  # same fallback for both

    # Other ratios
    bw_r = membw / REF["membw"]
    vr_r = vram / REF["vram"]
    ic_r = max(ic_bw / REF["ic_bw"], 0.001)

    # eTCU: compute-bound score
    eTCU = (
        fp_ratio_tcu ** TCU_EXP["fp"]
        * bw_r ** TCU_EXP["bw"]
        * vr_r ** TCU_EXP["vram"]
        * ic_r ** TCU_EXP["ic"]
    )

    # eICU: inference/memory-bound score
    eICU = (
        fp_ratio_icu ** ICU_EXP["fp"]
        * bw_r ** ICU_EXP["bw"]
        * vr_r ** ICU_EXP["vram"]
        * ic_r ** ICU_EXP["ic"]
    )

    # eCU: weighted harmonic mean (equal weights)
    eCU = 1.0 / (0.5 / eTCU + 0.5 / eICU)

    return {
        "eTCU": round(eTCU, 6),
        "eICU": round(eICU, 6),
        "eCU": round(eCU, 6),
        "fp8_fallback_used": fp8_fallback,
        "scoring_method": "geometric_mean_v4_harmonic_ecu",
        "reference_gpu": "8x_H100_SXM",
        "reference_values": REF,
        "v4_exponents": {"TCU": TCU_EXP, "ICU": ICU_EXP},
        "fp_weights": FP_W,
    }


def _seal_report(report: dict) -> None:
    """Compute and set integrity hash on report (in-place)."""
    payload = {k: v for k, v in report.items() if k not in INTEGRITY_FIELDS}
    canonical = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    report["integrity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report["integrity_method"] = (
        "sha256(json.dumps(report_without_integrity_fields, "
        "indent=2, sort_keys=True, ensure_ascii=True).encode('utf-8'))"
    )


def score_report(report_path: str) -> dict:
    """Read a benchmark report, compute eCU scores, append them, and re-seal.

    Args:
        report_path: path to benchmark_report.json

    Returns:
        The scored report dict.
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    measured = report.get("measured_specs", {})
    scores = compute_ecu_scores(measured)
    scores["scored_at"] = datetime.now(timezone.utc).isoformat()

    report["ecu_scores"] = scores

    # Re-seal integrity hash
    _seal_report(report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=True)

    return report


def main():
    """Score a benchmark report from RESULTS_DIR or argv[1]."""
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        results_dir = os.environ.get("RESULTS_DIR", "")
        if not results_dir:
            # Default: find most recent run in ./results/
            bm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_root = os.path.join(bm_root, "results")
            if os.path.isdir(results_root):
                runs = sorted(
                    [d for d in os.listdir(results_root) if d.startswith("run_")],
                    reverse=True,
                )
                if runs:
                    results_dir = os.path.join(results_root, runs[0])
        if not results_dir:
            print("ERROR: No RESULTS_DIR set and no runs found in ./results/")
            sys.exit(1)
        report_path = os.path.join(results_dir, "benchmark_report.json")

    if not os.path.exists(report_path):
        print(f"ERROR: Report not found: {report_path}")
        sys.exit(1)

    report = score_report(report_path)
    ecu = report["ecu_scores"]

    print(f"\n  eCU Scoring Engine (v4 exponents)")
    print(f"  {'─' * 40}")
    print(f"  Reference:  8x H100 SXM cluster (1 eCU)")
    print(f"  Report:     {report_path}")
    print(f"  {'─' * 40}")
    print(f"  eTCU:       {ecu['eTCU']:.6f}")
    print(f"  eICU:       {ecu['eICU']:.6f}")
    print(f"  eCU:        {ecu['eCU']:.6f}")
    if ecu["fp8_fallback_used"]:
        print(f"  [FP8 fallback: FP16 absorbed FP8 weight]")
    print(f"  {'─' * 40}")
    print(f"  Hash:       {report['integrity_sha256'][:32]}...")
    print()


if __name__ == "__main__":
    main()
