"""End-to-end pipeline test: report -> score -> verify."""
import json
import os
import shutil
import subprocess
import tempfile

BENCH_DIR = os.path.join(os.path.dirname(__file__), '..', 'bench')
RESULTS_SRC = os.path.join(os.path.dirname(__file__), '..', 'results', 'run_20260301T213834Z')
PYTHON = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python')


def test_score_on_real_report():
    """Run score.py on RTX 5090 data, verify eCU scores against 8x cluster ref."""
    if not os.path.isdir(RESULTS_SRC):
        import pytest
        pytest.skip("No existing benchmark results")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'benchmark_report.json')
        shutil.copy(
            os.path.join(RESULTS_SRC, 'benchmark_report.json'),
            report_path
        )

        result = subprocess.run(
            [PYTHON, os.path.join(BENCH_DIR, 'score.py'), report_path],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"score.py failed: {result.stderr}"
        assert "eTCU:" in result.stdout

        with open(report_path) as f:
            report = json.load(f)

        scores = report["ecu_scores"]
        assert 0.003 < scores["eTCU"] < 0.015, f"eTCU out of range: {scores['eTCU']}"
        assert 0.003 < scores["eICU"] < 0.015, f"eICU out of range: {scores['eICU']}"
        assert 0.003 < scores["eCU"] < 0.015, f"eCU out of range: {scores['eCU']}"
        assert scores["reference_gpu"] == "8x_H100_SXM"


def test_verify_report_after_scoring():
    """verify_report.py should pass on a scored report."""
    if not os.path.isdir(RESULTS_SRC):
        import pytest
        pytest.skip("No existing benchmark results")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'benchmark_report.json')
        shutil.copy(os.path.join(RESULTS_SRC, 'benchmark_report.json'), report_path)

        subprocess.run([PYTHON, os.path.join(BENCH_DIR, 'score.py'), report_path],
                       capture_output=True)

        result = subprocess.run(
            [PYTHON, os.path.join(BENCH_DIR, 'verify_report.py'), report_path],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"verify_report failed: {result.stdout}"
        assert "VERIFIED" in result.stdout
