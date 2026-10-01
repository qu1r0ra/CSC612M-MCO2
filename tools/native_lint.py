"""Static analysis for the C and CUDA sources: clang-tidy and MSVC /analyze.

Each translation unit is checked with the flags the justfile compiles it
with, read through `just --evaluate`, so a flag change in the justfile
reaches the lint without a second copy. `main.c` is checked with and without
SQ_ENABLE_CUDA because both builds compile it. Run inside the MSVC developer
environment (`tools/with-msvc.ps1`), which supplies the system headers.

    python tools/native_lint.py [--tidy] [--analyze]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stoquant.native_build import source_lists

REPO = Path(__file__).resolve().parents[1]


def c_units(root: Path = REPO) -> list[tuple[str, list[str]]]:
    """(source, MSVC flags) for every C translation unit the justfile compiles."""
    sources = source_lists(root)
    includes = _just_var(root, "cl_includes")
    host = _just_var(root, "cl_host_flags") + includes
    cuda_host = [*host, "/DSQ_ENABLE_CUDA"]
    test = _just_var(root, "cl_test_flags") + includes
    avx = _just_var(root, "avx2_cl_flags")
    units: list[tuple[str, list[str]]] = []

    def add(source: str, flags: list[str]) -> None:
        row = (source, flags)
        if source.endswith(".c") and row not in units:
            units.append(row)

    for source in sources["native_host_sources"]:
        add(source, host)
        add(source, cuda_host)
    for source in sources["native_avx2_sources"]:
        add(source, avx)
    for name in SOURCE_GROUPS_FOR_LINT:
        for source in sources[name]:
            add(source, avx if source in sources["native_avx2_sources"] else test)
    return units


SOURCE_GROUPS_FOR_LINT = (
    "native_rng_sources",
    "test_codec_sources",
    "test_quantizer_sources",
    "test_avx2_sources",
    "test_avx2_driver_sources",
    "test_rng_cpu_sources",
    "test_asan_probe_sources",
)


def _just_var(root: Path, name: str) -> list[str]:
    out = subprocess.run(
        ["just", "--evaluate", name], cwd=root, check=True, capture_output=True, text=True
    )
    return out.stdout.split()


def cuda_units(cuda_path: str, root: Path = REPO) -> list[tuple[str, list[str]]]:
    """(source, clang flags) for the CUDA translation units.

    Clang parses the whole file in device mode, host functions included.
    Random123 detects nvcc through __CUDACC__, which clang does not define, so
    its device qualifier and 64-bit multiply are set to what nvcc would select.
    """
    arch = _just_var(root, "cuda_arch")[0]
    if arch == "native":
        device = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        capability = device.stdout.splitlines()[0].strip()
        if not capability.replace(".", "").isdigit():
            raise RuntimeError(f"cannot resolve native CUDA architecture: {capability!r}")
        arch = "sm_" + capability.replace(".", "")
    flags = [
        "-xcuda",
        "--cuda-device-only",
        f"--cuda-gpu-arch={arch}",
        f"--cuda-path={cuda_path}",
        "-std=c++17",
        "-Wno-unknown-cuda-version",
        "-DR123_CUDA_DEVICE=__device__",
        "-DR123_USE_MULHILO64_CUDA_INTRIN=1",
        "-DR123_USE_MULHILO64_ASM=0",
        "-DR123_USE_MULHILO64_MSVC_INTRIN=0",
        "-DR123_USE_MULHILO64_C99=0",
        "-DR123_USE_MULHILO64_MULHI_INTRIN=0",
        "-DR123_USE_GNU_UINT128=0",
        *_just_var(root, "cc_includes"),
    ]
    sources = source_lists(root)
    cuda_sources = [
        source
        for name in ("native_cuda_sources", "native_probe_sources", "native_rng_sources")
        for source in sources[name]
        if source.endswith(".cu")
    ]
    return [(source, flags) for source in dict.fromkeys(cuda_sources)]


def run(cmd: list[str]) -> bool:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO, check=False).returncode == 0


def tidy() -> bool:
    exe = shutil.which("clang-tidy")
    if exe is None:
        sys.exit("clang-tidy not found; run through `uv run`")
    ok = True
    for source, flags in c_units():
        # clang-cl accepts the MSVC flags; optimisation and logo flags are irrelevant here.
        cl = [f for f in flags if f not in ("/nologo", "/O2")]
        ok &= run([exe, "--quiet", source, "--", "--driver-mode=cl", *cl])
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path is None:
        # Hosted CI runners have no CUDA toolkit; the CUDA files are checked locally.
        print("CUDA_PATH unset: skipping clang-tidy on .cu files", flush=True)
        return ok
    for source, flags in cuda_units(cuda_path):
        ok &= run([exe, "--quiet", source, "--", *flags])
    return ok


def analyze() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as out:
        for source, flags in c_units():
            # System headers are external and unanalyzed; C6993 only notes that
            # the analysis skips OpenMP regions.
            ok &= run(
                [
                    "cl.exe",
                    *flags,
                    "/analyze",
                    "/analyze:external-",
                    "/external:anglebrackets",
                    "/external:W0",
                    "/wd6993",
                    "/c",
                    source,
                    f"/Fo{out}\\",
                ]
            )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tidy", action="store_true", help="run clang-tidy only")
    parser.add_argument("--analyze", action="store_true", help="run MSVC /analyze only")
    args = parser.parse_args()
    both = not (args.tidy or args.analyze)
    ok = True
    if args.tidy or both:
        ok &= tidy()
    if args.analyze or both:
        ok &= analyze()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
