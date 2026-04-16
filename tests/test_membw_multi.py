"""Tests for multi-GPU parallel memory bandwidth benchmarking."""
import json
import os
import subprocess
import sys
import tempfile


PYTHON = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python')
MEMBW_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'membw.py')


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


def _run_membw(tmpdir, extra_env=None):
    """Run membw.py with small iters for speed, return parsed 02_membw.json."""
    env = os.environ.copy()
    env["RESULTS_DIR"] = tmpdir
    env["CU_MEMBW_ITERS"] = "10"
    env["CU_MEMBW_WARMUP"] = "3"
    env["_CU_BENCH_QUIET"] = "1"
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [PYTHON, MEMBW_SCRIPT],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, f"membw.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    out_path = os.path.join(tmpdir, "02_membw.json")
    assert os.path.exists(out_path), "02_membw.json not created"

    with open(out_path) as f:
        return json.load(f)


def test_single_gpu_has_gpu_count():
    """Single-GPU run should report gpu_count=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_membw(tmpdir)

        assert "gpu_count" in data, "gpu_count missing from output"
        assert data["gpu_count"] == 1


def test_single_gpu_has_per_gpu():
    """Single-GPU run should include per_gpu dict with one entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_membw(tmpdir)

        assert "per_gpu" in data, "per_gpu missing from output"
        per_gpu = data["per_gpu"]
        assert isinstance(per_gpu, dict), f"per_gpu should be dict, got {type(per_gpu)}"
        assert "gpu0" in per_gpu, "gpu0 missing from per_gpu"


def test_per_gpu_has_silicon_id():
    """Each per_gpu entry should have silicon_id from gpu_map."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_membw(tmpdir)

        gpu0 = data["per_gpu"]["gpu0"]
        assert "silicon_id" in gpu0, "silicon_id missing from per_gpu entry"
        assert gpu0["silicon_id"] == "GPU-FAKE-UUID-0000"


def test_per_gpu_has_test_results():
    """Each per_gpu entry should have test sub-dicts with gbps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_membw(tmpdir)

        gpu0 = data["per_gpu"]["gpu0"]
        # clone_primary should always be present
        assert "clone_primary" in gpu0, "clone_primary missing from per_gpu.gpu0"
        if gpu0["clone_primary"] is not None:
            assert "gbps" in gpu0["clone_primary"], "gbps missing from per_gpu.gpu0.clone_primary"


def test_cluster_gbps_in_top_level():
    """Top-level test results should include cluster_gbps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_env_json(tmpdir, gpu_count=1)
        data = _run_membw(tmpdir)

        r = data.get("clone_primary")
        if r is not None:
            assert "cluster_gbps" in r, "cluster_gbps missing from top-level clone_primary"
            # For single GPU, cluster_gbps == gbps
            assert r["cluster_gbps"] == r["gbps"], \
                f"Single GPU: cluster_gbps ({r['cluster_gbps']}) should equal gbps ({r['gbps']})"


def test_no_env_json_still_works():
    """membw.py should work even without 00_environment.json (backwards compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Don't create 00_environment.json
        data = _run_membw(tmpdir)

        assert "gpu_count" in data
        assert "per_gpu" in data


def test_aggregate_cluster_sums_gbps():
    """Test cluster aggregation logic without GPU hardware."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))
    from membw import _aggregate_cluster

    per_gpu = {
        "gpu0": {
            "silicon_id": "A",
            "clone_primary": {"gbps": 3200.0, "avg_ms": 1.0, "min_ms": 0.9, "max_ms": 1.1, "tensor_gb": 8.0, "bytes_moved": 16000000000},
            "clone_large": {"gbps": 3100.0, "avg_ms": 1.1, "min_ms": 1.0, "max_ms": 1.2, "tensor_gb": 20.0, "bytes_moved": 40000000000},
            "mul_primary": {"gbps": 3000.0, "avg_ms": 1.2, "method": "element_mul", "tensor_gb": 8.0, "bytes_moved": 24000000000},
        },
        "gpu1": {
            "silicon_id": "B",
            "clone_primary": {"gbps": 3190.0, "avg_ms": 1.01, "min_ms": 0.91, "max_ms": 1.11, "tensor_gb": 8.0, "bytes_moved": 16000000000},
            "clone_large": {"gbps": 3090.0, "avg_ms": 1.11, "min_ms": 1.01, "max_ms": 1.21, "tensor_gb": 20.0, "bytes_moved": 40000000000},
            "mul_primary": {"gbps": 2990.0, "avg_ms": 1.21, "method": "element_mul", "tensor_gb": 8.0, "bytes_moved": 24000000000},
        },
    }
    result = _aggregate_cluster(per_gpu, ["clone_primary", "clone_large", "mul_primary"])
    assert result["clone_primary"]["cluster_gbps"] == 6390.0
    assert result["clone_primary"]["gbps"] == 3195.0  # average
    assert result["clone_large"]["cluster_gbps"] == 6190.0
    assert result["mul_primary"]["cluster_gbps"] == 5990.0
