import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / ("mco2.exe" if os.name == "nt" else "mco2")

pytestmark = pytest.mark.skipif(
    os.environ.get("MCO2_EXPECT_CPU_ONLY") != "1",
    reason="run with `just test-cpu` to check the CPU-only executable",
)


def test_cpu_only_executable_rejects_cuda_backend(tmp_path):
    input_path = tmp_path / "input.f32"
    output_path = tmp_path / "record.msq"
    input_path.write_bytes(np.asarray([1.0], dtype="<f4").tobytes())

    result = subprocess.run(
        [
            str(BINARY),
            "compress",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            "1",
            "--backend",
            "cuda",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "built without CUDA support" in result.stderr
    assert not output_path.exists()


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_cpu_only_executable_rejects_gpu_origin_bench(tmp_path, backend):
    input_path = tmp_path / "input.f32"
    input_path.write_bytes(np.asarray([1.0], dtype="<f4").tobytes())

    result = subprocess.run(
        [
            str(BINARY),
            "bench",
            "--input",
            str(input_path),
            "--seed",
            "1",
            "--backend",
            backend,
            "--boundary",
            "gpu-origin",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "built without CUDA support" in result.stderr
    assert result.stdout == ""
