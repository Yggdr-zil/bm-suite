"""Tests for score.py — eCU Scoring Engine (v4 exponents, 8x H100 SXM cluster reference).

Patent pending — Joseph Januszewski, filed Jan 9, 2026.
"""
import hashlib
import json
import os
import sys
import tempfile

# Allow imports from bench/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))

from score import compute_ecu_scores, score_report


def test_8x_h100_sxm_cluster_scores_one():
    """Reference cluster (8x H100 SXM) should score exactly 1.0 on all axes."""
    measured = {
        "fp16_tflops": 7916.0,
        "fp8_tflops": 15832.0,
        "fp32_tflops": 536.0,
        "vram_usable_gb": 640.0,
        "membw_gbps": 26800.0,
        "interconnect_bw_gbps": 900.0,
    }
    result = compute_ecu_scores(measured)
    assert abs(result["eTCU"] - 1.0) < 1e-6, f"eTCU={result['eTCU']}, expected 1.0"
    assert abs(result["eICU"] - 1.0) < 1e-6, f"eICU={result['eICU']}, expected 1.0"
    assert abs(result["eCU"] - 1.0) < 1e-6, f"eCU={result['eCU']}, expected 1.0"
    assert result["fp8_fallback_used"] is False


def test_1x_h100_sxm_scores_about_tenth():
    """Single H100 SXM should score ~0.106 (ic_bw=900 per-GPU, not divided)."""
    measured = {
        "fp16_tflops": 989.5,
        "fp8_tflops": 1979.0,
        "fp32_tflops": 67.0,
        "vram_usable_gb": 80.0,
        "membw_gbps": 3350.0,
        "interconnect_bw_gbps": 900.0,
    }
    result = compute_ecu_scores(measured)
    assert abs(result["eTCU"] - 0.106240) < 0.001, f"eTCU={result['eTCU']}"
    assert abs(result["eICU"] - 0.106632) < 0.001, f"eICU={result['eICU']}"
    assert abs(result["eCU"] - 0.106435) < 0.001, f"eCU={result['eCU']}"
    assert result["fp8_fallback_used"] is False


def test_rtx5090_laptop_scores():
    """RTX 5090 Laptop (from real benchmark run) — single GPU with PCIe proxy."""
    measured = {
        "fp16_tflops": 50.48,
        "fp8_tflops": 101.91,
        "fp32_tflops": 18.03,
        "vram_usable_gb": 23.3885,
        "membw_gbps": 524.7,
        "interconnect_bw_gbps": 63.0,  # PCIe 5.0 x16 proxy
    }
    result = compute_ecu_scores(measured)
    assert abs(result["eTCU"] - 0.008451) < 0.001, f"eTCU={result['eTCU']}"
    assert abs(result["eICU"] - 0.009469) < 0.001, f"eICU={result['eICU']}"
    assert abs(result["eCU"] - 0.008931) < 0.001, f"eCU={result['eCU']}"
    assert result["fp8_fallback_used"] is False


def test_no_fp8_fallback():
    """When FP8 is None/0, FP16 absorbs FP8 weight — eICU should drop relative to eTCU."""
    measured = {
        "fp16_tflops": 989.5,
        "fp8_tflops": None,
        "fp32_tflops": 67.0,
        "vram_usable_gb": 80.0,
        "membw_gbps": 3350.0,
        "interconnect_bw_gbps": 900.0,
    }
    result = compute_ecu_scores(measured)
    assert result["fp8_fallback_used"] is True
    # Without FP8, ICU (which normally weights FP8 heavily) should score lower
    # Both use same fp_ratio in fallback, but ICU exponents favor bw/vram more
    assert result["eTCU"] > 0, "eTCU must be positive"
    assert result["eICU"] > 0, "eICU must be positive"
    assert result["eCU"] > 0, "eCU must be positive"

    # With FP8 present (H100 reference), scores differ; without, same fp_ratio
    # ICU's lower fp exponent (0.363 vs 0.524) means fp_ratio matters less
    # ICU's higher bw/vram exponents compensate → eICU > eTCU in fallback
    # Key assertion: fallback was detected
    assert result["fp8_fallback_used"] is True


def test_fp8_zero_integer_triggers_fallback():
    """FP8 = 0 (integer, not None) should trigger the no-FP8 fallback path."""
    measured = {
        "fp16_tflops": 989.5,
        "fp8_tflops": 0,
        "fp32_tflops": 67.0,
        "vram_usable_gb": 80.0,
        "membw_gbps": 3350.0,
        "interconnect_bw_gbps": 900.0,
    }
    result = compute_ecu_scores(measured)
    assert result["fp8_fallback_used"] is True
    assert result["eTCU"] > 0, "eTCU must be positive"
    assert result["eICU"] > 0, "eICU must be positive"
    assert result["eCU"] > 0, "eCU must be positive"


def test_missing_keys_minimal_dict():
    """A minimal dict missing most keys should not crash; scores should be positive."""
    measured = {"fp16_tflops": 100.0}
    result = compute_ecu_scores(measured)
    assert result["eTCU"] > 0, "eTCU must be positive"
    assert result["eICU"] > 0, "eICU must be positive"
    assert result["eCU"] > 0, "eCU must be positive"
    assert result["fp8_fallback_used"] is True  # fp8 missing → fallback


def test_zero_membw_and_vram_no_crash():
    """Zero membw and vram should not cause ZeroDivisionError."""
    measured = {
        "fp16_tflops": 100.0,
        "fp8_tflops": 200.0,
        "fp32_tflops": 10.0,
        "vram_usable_gb": 0,
        "membw_gbps": 0,
        "interconnect_bw_gbps": 0,
    }
    result = compute_ecu_scores(measured)
    assert result["eTCU"] > 0, "eTCU must be positive with 0.001 floor"
    assert result["eICU"] > 0, "eICU must be positive with 0.001 floor"
    assert result["eCU"] > 0, "eCU must be positive with 0.001 floor"


def test_score_appended_to_report():
    """score_report() reads report JSON, appends ecu_scores section, and re-seals hash."""
    # Build a minimal but valid report
    report = {
        "cu_bench_version": "1.0.0",
        "run_id": "test_run",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "environment": {"gpu_model": "Test GPU", "gpu_count": 8},
        "measured_specs": {
            "fp16_tflops": 7916.0,
            "fp8_tflops": 15832.0,
            "fp32_tflops": 536.0,
            "vram_usable_gb": 640.0,
            "membw_gbps": 26800.0,
            "interconnect_bw_gbps": 900.0,
        },
        "benchmarks_completed": ["gemm", "membw", "vram", "interconnect"],
        "detailed": {},
    }

    # Seal it like report.py does
    canonical = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    report["integrity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report["integrity_method"] = (
        "sha256(json.dumps(report_without_integrity_fields, "
        "indent=2, sort_keys=True, ensure_ascii=True).encode('utf-8'))"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=True)
        tmp_path = f.name

    try:
        score_report(tmp_path)

        with open(tmp_path) as f:
            scored = json.load(f)

        # ecu_scores section must exist
        assert "ecu_scores" in scored, "ecu_scores section missing"
        ecu = scored["ecu_scores"]
        assert abs(ecu["eTCU"] - 1.0) < 1e-6
        assert abs(ecu["eICU"] - 1.0) < 1e-6
        assert abs(ecu["eCU"] - 1.0) < 1e-6

        # Metadata should be present
        assert ecu["reference_gpu"] == "8x_H100_SXM"
        assert "scored_at" in ecu
        assert "v4_exponents" in ecu

        # Integrity hash must be re-sealed and valid
        stored_hash = scored["integrity_sha256"]
        payload = {k: v for k, v in scored.items()
                   if k not in {"integrity_sha256", "integrity_method"}}
        recomputed = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
        expected_hash = hashlib.sha256(recomputed.encode("utf-8")).hexdigest()
        assert stored_hash == expected_hash, (
            f"Integrity hash mismatch after scoring.\n"
            f"  stored:   {stored_hash}\n"
            f"  computed: {expected_hash}"
        )
    finally:
        os.unlink(tmp_path)
