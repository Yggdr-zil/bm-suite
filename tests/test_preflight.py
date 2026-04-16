"""Tests for preflight GPU map detection."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))


def test_gpu_map_structure():
    """Verify gpu_map in environment output has correct structure."""
    import subprocess
    import tempfile

    python = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python')

    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["RESULTS_DIR"] = tmpdir
        result = subprocess.run(
            [python, os.path.join(os.path.dirname(__file__), '..', 'bench', 'preflight.py')],
            capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, f"preflight failed: {result.stderr}"

        env_json_path = os.path.join(tmpdir, "00_environment.json")
        assert os.path.exists(env_json_path), "00_environment.json not created"

        with open(env_json_path) as f:
            env_data = json.load(f)

        gpu_map = env_data.get("gpu_map")
        assert gpu_map is not None, "gpu_map missing from environment"
        assert isinstance(gpu_map, dict), f"gpu_map should be dict, got {type(gpu_map)}"

        gpu_count = env_data.get("gpu_count", 1)
        assert len(gpu_map) == gpu_count, f"gpu_map has {len(gpu_map)} entries, expected {gpu_count}"

        gpu0 = gpu_map.get("gpu0")
        assert gpu0 is not None, "gpu0 missing from gpu_map"
        assert "index" in gpu0, "gpu0 missing 'index'"
        assert "uuid" in gpu0, "gpu0 missing 'uuid'"
        assert "name" in gpu0, "gpu0 missing 'name'"
        assert "serial" in gpu0, "gpu0 missing 'serial'"
        assert gpu0["index"] == 0, f"gpu0 index should be 0, got {gpu0['index']}"
