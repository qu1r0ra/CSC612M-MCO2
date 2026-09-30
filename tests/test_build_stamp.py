"""The binary locator refuses build facts it cannot verify."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stoquant.build_stamp import compiler_commands, recipe_commands, sha256, stamp_path
from stoquant.provenance import find_binary

TEST_JUSTFILE = """
cuda_arch := "native"
cl_includes := "/Inative"
cc_includes := "-Inative"
cl_host_flags := "/nologo /O2 /W4"
cc_host_flags := "-O2 -std=c11 -Wall -Wextra -Werror"
cl_test_flags := "/nologo /W4"
cc_strict_flags := "-std=c11 -Wall -Wextra -Werror"
avx2_cl_flags := "/nologo /O2 /arch:AVX2"
avx2_cc_flags := "-O2 -mavx2 -fopenmp"
nvcc_flags := "-O2 -arch=" + cuda_arch + " " + cc_includes
nvcc_warn_flags := "--Werror all-warnings"
nvcc_fp_flags := "--fmad=false"
native_host_sources := "native/main.c native/quantizer.c"
native_avx2_sources := "native/quantizer_avx2.c"
native_cuda_sources := "native/quantizer_cuda.cu"
native_rng_sources := "native/rng_cpu.c native/rng_cuda.cu tests/test_rng.c"
native_probe_sources := "native/stream_probe.cu"
test_codec_sources := "tests/test_codec.c native/codec.c"
test_quantizer_sources := "tests/test_quantizer.c native/quantizer.c native/codec.c native/rng_cpu.c"
test_avx2_sources := "tests/test_quantizer_avx2.c native/quantizer_avx2.c native/quantizer.c native/codec.c native/rng_cpu.c"
test_avx2_driver_sources := "tests/test_quantizer_avx2.c"
test_rng_cpu_sources := "tests/test_rng_cpu.c native/rng_cpu.c"
test_asan_probe_sources := "tests/asan_overread_probe.c"
"""


@pytest.fixture
def stamped_binary(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    (tmp_path / "justfile").write_text(TEST_JUSTFILE, encoding="utf-8")
    binary = tmp_path / "build" / ("stoquant.exe" if os.name == "nt" else "stoquant")
    binary.parent.mkdir()
    binary.write_bytes(b"verified binary")
    commands = recipe_commands(tmp_path, "build-cuda")
    stamp = {
        "version": 1,
        "revision": "a" * 40,
        "tree_dirty": False,
        "tree_fingerprint": "clean tree",
        "recipe": "build-cuda",
        "commands": commands,
        "compiler_flags": compiler_commands(commands),
        "binary_sha256": sha256(binary),
    }
    stamp_path(binary).write_text(json.dumps(stamp), encoding="utf-8")
    git = {
        "code_revision": "a" * 40,
        "git_dirty": False,
        "tree_fingerprint": "clean tree",
    }
    return tmp_path, binary, git


def test_locator_returns_verified_stamp(stamped_binary):
    root, binary, git = stamped_binary
    verified = find_binary(root, git_prov=git)
    assert verified.path == binary
    assert verified.stamp["binary_sha256"] == sha256(binary)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "build stamp missing"),
        ("stale", "revision is stale"),
        ("hash", "binary hash does not match"),
        ("cpu", "cannot serve CUDA paths"),
        ("fault", "test-only build define"),
    ],
)
def test_locator_refuses_unverifiable_build(stamped_binary, change, message):
    root, binary, git = stamped_binary
    path = stamp_path(binary)
    stamp = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        path.unlink()
    elif change == "stale":
        stamp["revision"] = "b" * 40
    elif change == "hash":
        binary.write_bytes(b"changed binary")
    else:
        recipe = "build-cpu" if change == "cpu" else "build-cuda-fault"
        commands = recipe_commands(root, recipe)
        stamp.update(
            recipe=recipe,
            commands=commands,
            compiler_flags=compiler_commands(commands),
        )
    if change not in ("missing", "hash"):
        path.write_text(json.dumps(stamp), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        find_binary(root, git_prov=git)


def test_stamp_detects_source_list_changes(stamped_binary):
    root, _, git = stamped_binary
    justfile = root / "justfile"
    justfile.write_text(
        justfile.read_text(encoding="utf-8").replace(
            'native_host_sources := "native/main.c native/quantizer.c"',
            'native_host_sources := "native/main.c native/quantizer.c native/new_unit.c"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="recipe flags are stale"):
        find_binary(root, git_prov=git)
