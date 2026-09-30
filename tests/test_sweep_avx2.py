import json
import os
from pathlib import Path

import numpy as np
import pytest
from _bench_test_support import READY_FACTS, ROOT, fake_stats

from stoquant.correctness import verify_correctness
from stoquant.design import (
    DEFAULT_TRIALS,
    build_paths,
    trial_design,
    trial_orders,
)
from stoquant.host import (
    EXCLUDED_LOGICAL_CPUS,
    affinity_mask_excluding,
    get_process_affinity,
)
from stoquant.matrix import run_benchmark_matrix
from stoquant.provenance import find_binary, parse_build_commands
from stoquant.runner import REAL_BENCH_PROCESS
from stoquant.stats import (
    compare_case_group,
)


def test_avx2_path_is_opt_in_and_leaves_the_default_design_unchanged():
    default = build_paths(["cpu", "cuda"])
    assert trial_orders(default, DEFAULT_TRIALS) == trial_orders(
        [path.label for path in default], DEFAULT_TRIALS
    )
    assert trial_design(len(default)) == "all-permutations"
    assert all(path.backend != "cpu-avx2" for path in default)

    paths = build_paths(["cpu", "cpu-avx2", "cuda"])
    assert [path.label for path in paths] == [
        "cpu-comparator",
        "cpu-avx2-optimized",
        "cuda-resident",
        "cuda-resident-graph",
        "cuda-host-origin",
    ]
    assert paths[1].extra_args == []
    assert paths[1].policy == "none"
    # Five paths need a Williams design, so the default 24 trials no longer divide.
    assert trial_design(len(paths)) == "williams"
    with pytest.raises(ValueError, match="multiple of 10"):
        trial_orders(paths, DEFAULT_TRIALS)
    assert [path.label for path in build_paths(["cpu", "cpu-avx2"])] == [
        "cpu-comparator",
        "cpu-avx2-optimized",
    ]


def test_avx2_path_is_descriptive_and_never_an_inversion():
    paths = build_paths(["cpu", "cpu-avx2", "cuda"])
    medians = [10.0, 2.0, 1.0, 0.8, 4.0]
    cases = [{"statistics": fake_stats(m)} for m in medians]
    compare_case_group(cases, paths)
    stats = {path.label: case["statistics"] for path, case in zip(paths, cases, strict=True)}

    avx2 = stats["cpu-avx2-optimized"]
    assert avx2["baseline"] == "cpu-comparator"
    assert avx2["speedup_vs_cpu"] == 5.0
    assert avx2["verdict"] == "faster"
    assert avx2["descriptive"] is True
    assert avx2["boundary_inversion"] is False
    for key in ("direction_supported", "magnitude_supported", "claim_supported_rev2"):
        assert avx2[key] is None
    # CUDA claims stay against the scalar comparator; the AVX2 ratio is descriptive.
    host = stats["cuda-host-origin"]
    assert host["baseline"] == "cpu-comparator"
    assert host["speedup_vs_cpu"] == 2.5
    assert host["direction_supported"] is True
    assert host["vs_cpu_avx2"]["speedup_vs_cpu_avx2"] == 0.5
    assert host["vs_cpu_avx2"]["verdict"] == "slower"
    assert host["vs_cpu_avx2"]["descriptive"] is True
    assert "direction_supported" not in host["vs_cpu_avx2"]
    assert "vs_cpu_avx2" not in stats["cpu-comparator"]


def test_build_flags_are_selected_by_source_file():
    commands = [
        "New-Item -ItemType Directory -Force build | Out-Null",
        r"./tools/with-msvc.ps1 cl.exe /O2 /fp:strict /DSQ_ENABLE_CUDA /c native\main.c /Fo:x",
        r"./tools/with-msvc.ps1 cl.exe /O2 /fp:strict /DSQ_ENABLE_CUDA /c native\quantizer.c",
        r"./tools/with-msvc.ps1 cl.exe /O2 /fp:precise /arch:AVX2 /openmp /c native\quantizer_avx2.c",
        r"./tools/with-msvc.ps1 nvcc -O3 --fmad=false -c native\quantizer_cuda.cu -o build\k.obj",
    ]
    flags = parse_build_commands(commands)
    assert flags.comparator_c == "/O2 /fp:strict /DSQ_ENABLE_CUDA"
    assert flags.avx2_c == "/O2 /fp:precise /arch:AVX2 /openmp"
    assert flags.cuda_nvcc == "-O3 --fmad=false"
    assert flags.host_tokens == ["/O2", "/fp:strict", "/DSQ_ENABLE_CUDA"]
    assert flags.avx2_tokens == ["/O2", "/fp:precise", "/arch:AVX2", "/openmp"]
    assert "host_tokens" not in flags.as_record()
    assert parse_build_commands([]).avx2_c == "unknown"


def test_correctness_gate_checks_the_avx2_path_without_cuda(tmp_path):
    values = np.linspace(-3.0, 3.0, 1287, dtype=np.float32)
    input_path = tmp_path / "input.f32"
    input_path.write_bytes(values.astype("<f4").tobytes())
    backends = ["cpu", "cpu-avx2"]
    passed, info = verify_correctness(
        find_binary(ROOT, require_cuda=False).path,
        input_path,
        len(values),
        4,
        backends=backends,
        paths=build_paths(backends),
        tmp_dir=tmp_path,
    )
    assert passed, info
    assert info["byte_identical_to_compress"] is True


def test_an_avx2_mismatch_is_not_reported_as_a_cpu_cuda_divergence(tmp_path):
    values = np.linspace(-3.0, 3.0, 1287, dtype=np.float32)
    input_path = tmp_path / "input.f32"
    input_path.write_bytes(values.astype("<f4").tobytes())

    class CorruptingRecordProcess:
        def run(self, argv, *, creationflags=0):
            result = REAL_BENCH_PROCESS.run(argv, creationflags=creationflags)
            if "compress" in argv and argv[-1] == "cpu-avx2":
                out = Path(argv[argv.index("--output") + 1])
                data = bytearray(out.read_bytes())
                data[-1] ^= 1
                out.write_bytes(bytes(data))
            return result

    backends = ["cpu", "cpu-avx2"]
    passed, info = verify_correctness(
        find_binary(ROOT, require_cuda=False).path,
        input_path,
        len(values),
        4,
        backends=backends,
        paths=build_paths(backends),
        tmp_dir=tmp_path,
        process=CorruptingRecordProcess(),
    )
    assert not passed
    assert info["cpu_avx2_byte_identical"] is False
    assert "cpu_cuda_byte_identical" not in info


def test_driver_cpu_avx2_snapshot_records_threads_and_flags(tmp_path):
    snapshot_dir = tmp_path / "avx2-snapshot"
    run_benchmark_matrix(
        root=ROOT,
        output_dir=snapshot_dir,
        counts=[1024],
        bit_widths=[8],
        backends=["cpu", "cpu-avx2"],
        warmups=1,
        reps=2,
        trials=2,
        in_process_warmup_seconds=0,
        allow_dirty=True,
        readiness_facts=READY_FACTS,
    )
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["matrix_parameters"]["paths"] == ["cpu-comparator", "cpu-avx2-optimized"]
    assert manifest["all_cases_passed"] is True
    flags = manifest["build_flags"]
    assert "/fp:strict" in flags["comparator_c"]
    assert "/arch:AVX2" not in flags["comparator_c"]
    assert "/arch:AVX2" in flags["avx2_c"]
    assert "/openmp" in flags["avx2_c"]
    mask = affinity_mask_excluding(EXCLUDED_LOGICAL_CPUS, get_process_affinity())
    expected_threads = min(mask.bit_count(), 256)
    assert manifest["matrix_parameters"]["cpu_avx2_threads"] == [expected_threads]

    case = json.loads(
        (snapshot_dir / "case_cpu-avx2_optimized_bits8_n1024.json").read_text(encoding="utf-8")
    )
    assert case["threads"] == expected_threads
    assert all(run["threads"] == expected_threads for run in case["trial_runs"])
    assert case["build_flags"]["avx2_c"] == flags["avx2_c"]
    assert case["statistics"]["descriptive"] is True
    if os.name == "nt":
        report = (snapshot_dir / "msvc_vectorization_report_avx2.txt").read_text(encoding="utf-8")
        assert "quantizer_avx2.c" in report
