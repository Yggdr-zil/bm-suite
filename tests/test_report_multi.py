"""Tests for report.py multi-GPU cluster aggregate extraction."""
import json
import os
import subprocess
import tempfile


PYTHON = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python')
REPORT_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'report.py')


def _write_json(tmpdir, filename, data):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _make_multi_gpu_fixtures(tmpdir, gpu_count=2):
    """Write fake multi-GPU benchmark JSONs to tmpdir."""
    _write_json(tmpdir, "00_environment.json", {
        "gpu_model": "NVIDIA FakeGPU 80GB",
        "gpu_count": gpu_count,
        "gpu_map": {
            f"gpu{i}": {"index": i, "uuid": f"GPU-FAKE-{i:04d}", "name": "NVIDIA FakeGPU 80GB"}
            for i in range(gpu_count)
        },
        "platform": "Linux-test",
        "driver_version": "555.42",
        "pytorch_version": "2.3.0",
    })

    _write_json(tmpdir, "01_gemm.json", {
        "device": "NVIDIA FakeGPU 80GB",
        "gpu_count": gpu_count,
        "fp32": {"tflops": 33.5, "cluster_tflops": 67.0, "avg_ms": 1.0},
        "fp16": {"tflops": 150.0, "cluster_tflops": 300.0, "avg_ms": 0.5},
        "bf16": {"tflops": 148.0, "cluster_tflops": 296.0, "avg_ms": 0.51},
        "fp8": {"tflops": 290.0, "cluster_tflops": 580.0, "avg_ms": 0.26},
    })

    _write_json(tmpdir, "02_membw.json", {
        "clone_large": {"gbps": 1600.0, "cluster_gbps": 3200.0, "size_gb": 4.0},
    })

    _write_json(tmpdir, "03_vram.json", {
        "total_verified_gb": 160.0,
        "total_reported_gb": 160.0,
        "gpus": {
            "gpu0": {"verified_total_gb": 80.0},
            "gpu1": {"verified_total_gb": 80.0},
        },
    })

    _write_json(tmpdir, "04_interconnect.json", {
        "peak_busbw_gbps": 450.0,
        "skipped": False,
    })


def _make_single_gpu_fixtures(tmpdir):
    """Write fake single-GPU benchmark JSONs (no cluster_* fields)."""
    _write_json(tmpdir, "00_environment.json", {
        "gpu_model": "NVIDIA FakeGPU 80GB",
        "gpu_count": 1,
        "platform": "Linux-test",
        "driver_version": "555.42",
        "pytorch_version": "2.3.0",
    })

    _write_json(tmpdir, "01_gemm.json", {
        "device": "NVIDIA FakeGPU 80GB",
        "gpu_count": 1,
        "fp32": {"tflops": 33.5, "avg_ms": 1.0},
        "fp16": {"tflops": 150.0, "avg_ms": 0.5},
        "bf16": {"tflops": 148.0, "avg_ms": 0.51},
        "fp8": None,
    })

    _write_json(tmpdir, "02_membw.json", {
        "clone_large": {"gbps": 1600.0, "size_gb": 4.0},
    })

    _write_json(tmpdir, "03_vram.json", {
        "total_verified_gb": 80.0,
        "total_reported_gb": 80.0,
        "gpus": {"gpu0": {"verified_total_gb": 80.0}},
    })

    _write_json(tmpdir, "04_interconnect.json", {
        "skipped": True,
        "reason": "single_gpu",
    })


def _run_report(tmpdir):
    """Run report.py against tmpdir, return parsed benchmark_report.json."""
    env = os.environ.copy()
    env["RESULTS_DIR"] = tmpdir
    env["_CU_BENCH_QUIET"] = "1"

    result = subprocess.run(
        [PYTHON, REPORT_SCRIPT],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"report.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    report_path = os.path.join(tmpdir, "benchmark_report.json")
    assert os.path.exists(report_path), "benchmark_report.json not created"
    with open(report_path) as f:
        return json.load(f)


def test_multi_gpu_uses_cluster_tflops():
    """Multi-GPU: measured_specs should use cluster_tflops, not per-GPU tflops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multi_gpu_fixtures(tmpdir, gpu_count=2)
        report = _run_report(tmpdir)
        m = report["measured_specs"]

        assert m["fp32_tflops"] == 67.0, f"Expected cluster 67.0, got {m['fp32_tflops']}"
        assert m["fp16_tflops"] == 300.0, f"Expected cluster 300.0, got {m['fp16_tflops']}"
        assert m["bf16_tflops"] == 296.0, f"Expected cluster 296.0, got {m['bf16_tflops']}"
        assert m["fp8_tflops"] == 580.0, f"Expected cluster 580.0, got {m['fp8_tflops']}"


def test_multi_gpu_uses_cluster_gbps():
    """Multi-GPU: measured_specs should use cluster_gbps for membw."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multi_gpu_fixtures(tmpdir, gpu_count=2)
        report = _run_report(tmpdir)
        m = report["measured_specs"]

        assert m["membw_gbps"] == 3200.0, f"Expected cluster 3200.0, got {m['membw_gbps']}"


def test_multi_gpu_has_gpu_count():
    """Multi-GPU: measured_specs should include gpu_count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multi_gpu_fixtures(tmpdir, gpu_count=2)
        report = _run_report(tmpdir)
        m = report["measured_specs"]

        assert "gpu_count" in m, "gpu_count missing from measured_specs"
        assert m["gpu_count"] == 2


def test_single_gpu_fallback():
    """Single-GPU (no cluster_* fields): should fall back to per-GPU values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_single_gpu_fixtures(tmpdir)
        report = _run_report(tmpdir)
        m = report["measured_specs"]

        assert m["gpu_count"] == 1
        assert m["fp32_tflops"] == 33.5
        assert m["fp16_tflops"] == 150.0
        assert m["bf16_tflops"] == 148.0
        assert m["fp8_tflops"] is None  # fp8 was None in fixture
        assert m["membw_gbps"] == 1600.0
