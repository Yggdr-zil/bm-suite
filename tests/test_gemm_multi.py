"""Tests for multi-GPU parallel GEMM benchmarking."""
import json
import os
import subprocess
import sys
import tempfile


PYTHON = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python')
GEMM_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'gemm.py')


def _make_env_json(tmpdir, gpu_count=1):
    """Write a fake 00_environment.json with gpu_map to tmpdir."""
    gpu_map = {}
    for i in range(gpu_count):
        gpu_map[f"gpu{i}"] = {
            "index": i,
            "uuid": f"GPU-FAKE-UUID-{i:04d}",
            "name": "NVIDIA FakeGPU 80GB",
            "serial": f"SERIAL{i:04d}",
        }
    env_data = {
        "gpu_count": gpu_count,
        "gpu_map": gpu_map,
        "device": "NVIDIA FakeGPU 80GB",
    }
    path = os.path.join(tmpdir, "00_environment.json")
    with open(path, "w") as f:
        json.dump(env_data, f, indent=2)
    return path


def _run_gemm(tmpdir, extra_env=None):
    """Run gemm.py with small dims for speed, return parsed 01_gemm.json."""
    env = os.environ.copy()
    env["RESULTS_DIR"] = tmpdir
    env["CU_GEMM_DIM"] = "2048"
    env["CU_GEMM_ITERS"] = "10"
    env["CU_GEMM_WARMUP"] = "5"
    env["_CU_BENCH_QUIET"] = "1"
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [PYTHON, GEMM_SCRIPT],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, f"gemm.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    out_path = os.path.join(tmpdir, "01_gemm.json")
    assert os.path.exists(out_path), "01_gemm.json not created"

    with open(out_path) as f:
        return json.load(f)


def test_single_gpu_has_gpu_count():
    """Single-GPU run should report gpu_count=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_gemm(tmpdir)

        assert "gpu_count" in data, "gpu_count missing from output"
        assert data["gpu_count"] == 1


def test_single_gpu_has_per_gpu():
    """Single-GPU run should include per_gpu dict with one entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_gemm(tmpdir)

        assert "per_gpu" in data, "per_gpu missing from output"
        per_gpu = data["per_gpu"]
        assert isinstance(per_gpu, dict), f"per_gpu should be dict, got {type(per_gpu)}"
        assert "gpu0" in per_gpu, "gpu0 missing from per_gpu"


def test_per_gpu_has_silicon_id():
    """Each per_gpu entry should have silicon_id from gpu_map."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_gemm(tmpdir)

        gpu0 = data["per_gpu"]["gpu0"]
        assert "silicon_id" in gpu0, "silicon_id missing from per_gpu entry"
        assert gpu0["silicon_id"] == "GPU-FAKE-UUID-0000"


def test_per_gpu_has_precision_results():
    """Each per_gpu entry should have precision sub-dicts with tflops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_gemm(tmpdir)

        gpu0 = data["per_gpu"]["gpu0"]
        # At minimum fp16 and fp32 should be present (bf16 and fp8 may be None/skipped)
        for prec in ("fp16", "fp32"):
            assert prec in gpu0, f"{prec} missing from per_gpu.gpu0"
            if gpu0[prec] is not None:
                assert "tflops" in gpu0[prec], f"tflops missing from per_gpu.gpu0.{prec}"


def test_cluster_tflops_in_top_level():
    """Top-level precision results should include cluster_tflops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_gemm(tmpdir)

        for prec in ("fp16", "fp32"):
            r = data.get(prec)
            if r is not None:
                assert "cluster_tflops" in r, f"cluster_tflops missing from top-level {prec}"
                # For single GPU, cluster_tflops == tflops
                assert r["cluster_tflops"] == r["tflops"], \
                    f"Single GPU: cluster_tflops ({r['cluster_tflops']}) should equal tflops ({r['tflops']})"


def test_no_env_json_still_works():
    """gemm.py should work even without 00_environment.json (backwards compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Don't create 00_environment.json
        data = _run_gemm(tmpdir)

        # Should still have gpu_count (defaults to torch.cuda.device_count())
        assert "gpu_count" in data
        # per_gpu should still exist
        assert "per_gpu" in data


def test_aggregate_cluster_sums_tflops():
    """Test cluster aggregation logic without GPU hardware."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))
    from gemm import _aggregate_cluster

    per_gpu = {
        "gpu0": {"silicon_id": "A", "fp16": {"tflops": 100.0, "avg_ms": 1.0, "median_ms": 1.0, "min_ms": 0.9, "max_ms": 1.1, "std_ms": 0.05, "cv_pct": 5.0, "raw_avg_ms": 1.0, "trimmed_samples": 190, "total_samples": 200, "trim_pct": 5.0, "matrix_dim": 8192, "warmup_iters": 20, "bench_iters": 200}},
        "gpu1": {"silicon_id": "B", "fp16": {"tflops": 98.0, "avg_ms": 1.02, "median_ms": 1.02, "min_ms": 0.92, "max_ms": 1.12, "std_ms": 0.05, "cv_pct": 5.0, "raw_avg_ms": 1.02, "trimmed_samples": 190, "total_samples": 200, "trim_pct": 5.0, "matrix_dim": 8192, "warmup_iters": 20, "bench_iters": 200}},
    }
    result = _aggregate_cluster(per_gpu, ["fp16", "fp32", "bf16", "fp8"])
    assert result["fp16"]["cluster_tflops"] == 198.0
    assert result["fp16"]["tflops"] == 99.0  # average
