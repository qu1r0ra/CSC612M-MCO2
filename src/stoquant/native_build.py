"""Build native targets from the source and flag variables in the justfile."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

SOURCE_LIST_VARIABLES = (
    "native_host_sources",
    "native_avx2_sources",
    "native_cuda_sources",
    "native_rng_sources",
    "native_probe_sources",
    "test_cuda_api_sources",
    "test_codec_sources",
    "test_quantizer_sources",
    "test_avx2_sources",
    "test_avx2_driver_sources",
    "test_rng_cpu_sources",
    "test_asan_probe_sources",
)
TEST_TARGETS = (
    ("build-codec-test", "test_codec_sources", "build/test_codec"),
    ("build-quantizer-test", "test_quantizer_sources", "build/test_quantizer"),
    ("build-avx2-test", "test_avx2_sources", "build/test_quantizer_avx2"),
    ("build-rng-cpu-test", "test_rng_cpu_sources", "build/test_rng_cpu"),
)
BUILD_RECIPES = (
    "build-cpu",
    "build-cuda",
    "build-cuda-fault",
    "build-cuda-api-test",
    "build-stream-probe",
    "build-rng",
    *(row[0] for row in TEST_TARGETS),
    "test-cpu-gcc-ubsan",
)


def just_var(root: Path, name: str) -> list[str]:
    result = subprocess.run(
        ["just", "--evaluate", name], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot evaluate justfile variable {name}: {result.stderr.strip()}")
    return result.stdout.split()


def source_lists(root: Path) -> dict[str, list[str]]:
    return {name: just_var(root, name) for name in SOURCE_LIST_VARIABLES}


def _is_msvc() -> bool:
    return os.name == "nt"


def _path(root: Path, relative: str) -> Path:
    return root / relative.replace("/", os.sep)


def _object_path(group: str, source: str, extension: str) -> str:
    stem = Path(source).stem
    return f"build/{group}/{stem}.{extension}"


def _compiler(root: Path, *, force_gcc: bool = False) -> str:
    if force_gcc:
        return "gcc"
    if _is_msvc():
        return "cl.exe"
    return os.environ.get("CC", "cc")


def _flags(root: Path, name: str) -> list[str]:
    return just_var(root, name)


def _compile_objects(
    root: Path,
    sources: Sequence[str],
    *,
    group: str,
    compiler: str,
    flags: Sequence[str],
    includes: Sequence[str] = (),
    defines: Sequence[str] = (),
    msvc: bool | None = None,
) -> tuple[list[list[str]], list[str]]:
    use_msvc = _is_msvc() if msvc is None else msvc
    commands: list[list[str]] = []
    objects: list[str] = []
    extension = "obj" if use_msvc else "o"
    for source in sources:
        output = _object_path(group, source, extension)
        objects.append(output)
        if use_msvc:
            command = [compiler, *flags, *defines, *includes, "/c", source, f"/Fo:{output}"]
        else:
            command = [compiler, *flags, *defines, *includes, "-c", source, "-o", output]
        commands.append(command)
    return commands, objects


def _cpu_plan(
    root: Path,
    *,
    force_gcc: bool = False,
    sanitizer: Sequence[str] = (),
    binary: str | None = None,
) -> tuple[list[list[str]], str]:
    sources = source_lists(root)
    use_msvc = _is_msvc() and not force_gcc
    compiler = _compiler(root, force_gcc=force_gcc)
    if use_msvc:
        host_flags = _flags(root, "cl_host_flags")
        includes = _flags(root, "cl_includes")
        avx_flags = _flags(root, "avx2_cl_flags")
    else:
        host_flags = _flags(root, "cc_host_flags")
        includes = _flags(root, "cc_includes")
        avx_flags = _flags(root, "avx2_cc_flags")
    commands, host_objects = _compile_objects(
        root,
        sources["native_host_sources"],
        group="cpu",
        compiler=compiler,
        flags=[*host_flags, *sanitizer],
        includes=includes,
        msvc=use_msvc,
    )
    avx_commands, avx_objects = _compile_objects(
        root,
        sources["native_avx2_sources"],
        group="cpu",
        compiler=compiler,
        flags=[*avx_flags, *sanitizer],
        msvc=use_msvc,
    )
    commands.extend(avx_commands)
    binary = binary or ("build/stoquant.exe" if use_msvc else "build/stoquant")
    if use_msvc:
        commands.append([compiler, "/nologo", *host_objects, *avx_objects, f"/Fe:{binary}"])
    else:
        commands.append(
            [compiler, *sanitizer, "-fopenmp", *host_objects, *avx_objects, "-lm", "-o", binary]
        )
    return commands, binary


def _cuda_plan(root: Path) -> tuple[list[list[str]], str]:
    sources = source_lists(root)
    commands: list[list[str]] = []
    compiler = _compiler(root)
    cuda_host_define = ["/DSQ_ENABLE_CUDA"] if _is_msvc() else ["-DSQ_ENABLE_CUDA"]
    host_flags = _flags(root, "cl_host_flags" if _is_msvc() else "cc_host_flags")
    host_includes = _flags(root, "cl_includes" if _is_msvc() else "cc_includes")
    avx_flags = _flags(root, "avx2_cl_flags" if _is_msvc() else "avx2_cc_flags")
    host_commands, host_objects = _compile_objects(
        root,
        sources["native_host_sources"],
        group="cuda",
        compiler=compiler,
        flags=host_flags,
        includes=host_includes,
        defines=cuda_host_define,
    )
    avx_commands, avx_objects = _compile_objects(
        root,
        sources["native_avx2_sources"],
        group="cuda",
        compiler=compiler,
        flags=avx_flags,
    )
    commands.extend(host_commands)
    commands.extend(avx_commands)

    nvcc_flags = _flags(root, "nvcc_flags")
    nvcc_warn_flags = _flags(root, "nvcc_warn_flags")
    nvcc_fp_flags = _flags(root, "nvcc_fp_flags")
    cuda_objects: list[str] = []
    for source in sources["native_cuda_sources"]:
        output = _object_path("cuda", source, "obj" if _is_msvc() else "o")
        cuda_objects.append(output)
        commands.append(
            ["nvcc", *nvcc_flags, *nvcc_fp_flags, *nvcc_warn_flags, "-c", source, "-o", output]
        )
    binary = "build/stoquant.exe" if _is_msvc() else "build/stoquant"
    link_inputs = host_objects + avx_objects + cuda_objects
    if _is_msvc():
        commands.append(["nvcc", *nvcc_flags, *link_inputs, "-o", binary])
    else:
        commands.append(
            ["nvcc", *nvcc_flags, *link_inputs, "-lm", "-Xcompiler", "-fopenmp", "-o", binary]
        )
    return commands, binary


def _cuda_fault_plan(root: Path) -> tuple[list[list[str]], str]:
    """Build the CUDA fault and host-memory inspection test binary."""
    sources = source_lists(root)
    nvcc_flags = _flags(root, "nvcc_flags")
    nvcc_fp_flags = _flags(root, "nvcc_fp_flags")
    nvcc_warn_flags = _flags(root, "nvcc_warn_flags")
    extension = "obj" if _is_msvc() else "o"
    cuda_objects: list[str] = []
    commands: list[list[str]] = []
    for source in sources["native_cuda_sources"]:
        output = _object_path("cuda", f"{Path(source).stem}_fault.cu", extension)
        cuda_objects.append(output)
        commands.append(
            [
                "nvcc",
                *nvcc_flags,
                *nvcc_fp_flags,
                *nvcc_warn_flags,
                "-DSQ_CUDA_FAULT_INJECTION",
                "-DSQ_CUDA_BUFFER_INSPECTION",
                "-c",
                source,
                "-o",
                output,
            ]
        )
    host_objects = [
        _object_path("cuda", source, extension) for source in sources["native_host_sources"]
    ]
    avx_objects = [
        _object_path("cuda", source, extension) for source in sources["native_avx2_sources"]
    ]
    binary = "build/stoquant_fault.exe" if _is_msvc() else "build/stoquant_fault"
    link_inputs = host_objects + avx_objects + cuda_objects
    if _is_msvc():
        commands.append(["nvcc", *nvcc_flags, *link_inputs, "-o", binary])
    else:
        commands.append(
            ["nvcc", *nvcc_flags, *link_inputs, "-lm", "-Xcompiler", "-fopenmp", "-o", binary]
        )
    return commands, binary


def _cuda_api_test_plan(root: Path) -> tuple[list[list[str]], str]:
    sources = source_lists(root)
    nvcc_flags = _flags(root, "nvcc_flags")
    nvcc_fp_flags = _flags(root, "nvcc_fp_flags")
    nvcc_warn_flags = _flags(root, "nvcc_warn_flags")
    test_sources = sources["test_cuda_api_sources"]
    if not test_sources:
        raise RuntimeError("test_cuda_api_sources is empty")
    binary = "build/test_cuda_api.exe" if _is_msvc() else "build/test_cuda_api"
    command = [
        "nvcc",
        *nvcc_flags,
        *nvcc_fp_flags,
        *nvcc_warn_flags,
        *test_sources,
        *sources["native_cuda_sources"],
        "-o",
        binary,
    ]
    return [command], binary


def _probe_plan(root: Path) -> tuple[list[list[str]], str]:
    sources = source_lists(root)["native_probe_sources"]
    if not sources:
        raise RuntimeError("native_probe_sources is empty")
    nvcc_flags = _flags(root, "nvcc_flags")
    warnings = _flags(root, "nvcc_warn_flags")
    binary = "build/stream_probe.exe" if _is_msvc() else "build/stream_probe"
    return [["nvcc", *nvcc_flags, *warnings, *sources, "-o", binary]], binary


def _rng_plan(root: Path) -> tuple[list[list[str]], str]:
    sources = source_lists(root)["native_rng_sources"]
    if not sources:
        raise RuntimeError("rng_sources is empty")
    nvcc_flags = _flags(root, "nvcc_flags")
    warnings = _flags(root, "nvcc_warn_flags")
    binary = "build/test_rng.exe" if _is_msvc() else "build/test_rng"
    return [["nvcc", *nvcc_flags, *warnings, *sources, "-o", binary]], binary


def _test_plan(
    root: Path,
    recipe: str,
    *,
    extra_flags: Sequence[str] = (),
    output_override: str | None = None,
    force_gcc: bool = False,
) -> tuple[list[list[str]], str]:
    match = next((row for row in TEST_TARGETS if row[0] == recipe), None)
    if match is None:
        raise RuntimeError(f"unsupported native build recipe: {recipe}")
    _, source_variable, binary = match
    binary = output_override or binary
    sources = just_var(root, source_variable)
    use_msvc = _is_msvc() and not force_gcc
    compiler = _compiler(root, force_gcc=force_gcc)
    if recipe != "build-avx2-test":
        if use_msvc:
            flags = _flags(root, "cl_test_flags") + _flags(root, "cl_includes")
            output = f"{binary}.exe"
            return [["cl.exe", *flags, *extra_flags, *sources, f"/Fe:{output}"]], output
        flags = _flags(root, "cc_strict_flags") + _flags(root, "cc_includes")
        return [[compiler, *flags, *extra_flags, *sources, "-lm", "-o", binary]], binary

    commands: list[list[str]] = []
    objects: list[str] = []
    for source in sources:
        if source in source_lists(root)["native_avx2_sources"]:
            flags = _flags(root, "avx2_cl_flags" if use_msvc else "avx2_cc_flags")
        elif source in source_lists(root)["test_avx2_driver_sources"]:
            flags = _flags(root, "cl_test_flags" if use_msvc else "cc_strict_flags")
            if use_msvc:
                flags += _flags(root, "cl_includes")
            else:
                flags += _flags(root, "cc_includes")
        else:
            flags = _flags(root, "cl_host_flags" if use_msvc else "cc_host_flags")
            if use_msvc:
                flags += _flags(root, "cl_includes")
            else:
                flags += _flags(root, "cc_includes")
                flags += ["-fopenmp"]
        flags = [*flags, *extra_flags]
        extension = "obj" if use_msvc else "o"
        output = _object_path("test-avx2", source, extension)
        objects.append(output)
        if use_msvc:
            commands.append(["cl.exe", *flags, "/c", source, f"/Fo:{output}"])
        else:
            commands.append([compiler, *flags, "-c", source, "-o", output])
    output = f"{binary}.exe" if use_msvc else binary
    if use_msvc:
        commands.append(["cl.exe", "/nologo", *extra_flags, *objects, f"/Fe:{output}"])
    else:
        commands.append([compiler, *extra_flags, "-fopenmp", *objects, "-lm", "-o", output])
    return commands, output


def _ubsan_plan(root: Path) -> tuple[list[list[str]], list[str]]:
    sanitizer = ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
    cpu, _ = _cpu_plan(
        root,
        force_gcc=True,
        sanitizer=sanitizer,
        binary="build/stoquant_gcc_ubsan",
    )
    commands = list(cpu)
    binaries: list[str] = []
    for recipe, _, output in TEST_TARGETS:
        test_binary = f"{output}_ubsan"
        plan, _ = _test_plan(
            root,
            recipe,
            extra_flags=sanitizer,
            output_override=test_binary,
            force_gcc=True,
        )
        commands.extend(plan)
        binaries.append(test_binary)
    return commands, binaries


def command_plan(root: Path, recipe: str) -> tuple[list[list[str]], str]:
    """Return compiler argv arrays from the same evaluated lists the build uses."""
    if recipe == "build-cpu":
        return _cpu_plan(root)
    if recipe == "build-cuda":
        return _cuda_plan(root)
    if recipe == "build-cuda-fault":
        normal_commands, _ = _cuda_plan(root)
        fault_commands, binary = _cuda_fault_plan(root)
        return [*normal_commands, *fault_commands], binary
    if recipe == "build-cuda-api-test":
        return _cuda_api_test_plan(root)
    if recipe == "build-stream-probe":
        return _probe_plan(root)
    if recipe == "build-rng":
        return _rng_plan(root)
    if recipe == "test-cpu-gcc-ubsan":
        commands, _ = _ubsan_plan(root)
        return commands, ""
    return _test_plan(root, recipe)


def render_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command)) if _is_msvc() else shlex.join(command)


def recipe_commands(root: Path, recipe: str) -> list[str]:
    commands, _ = command_plan(root, recipe)
    return [render_command(command) for command in commands]


def _run_command(root: Path, command: Sequence[str]) -> None:
    if _is_msvc() and command[0] in {"cl.exe", "nvcc"}:
        wrapped = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "tools" / "with-msvc.ps1"),
            *command,
        ]
    else:
        wrapped = list(command)
    print("+", render_command(command), flush=True)
    result = subprocess.run(wrapped, cwd=root, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"native build command failed with exit {result.returncode}")


def _run_test_suite(root: Path) -> None:
    commands, binaries = _ubsan_plan(root)
    (root / "build").mkdir(parents=True, exist_ok=True)
    (root / "build" / "cpu").mkdir(parents=True, exist_ok=True)
    (root / "build" / "test-avx2").mkdir(parents=True, exist_ok=True)
    for command in commands:
        _run_command(root, command)
    for binary in binaries:
        _run_command(root, [str(_path(root, binary))])


def _run_asan_suite(root: Path) -> None:
    if not _is_msvc():
        raise RuntimeError("the AddressSanitizer CPU proof uses MSVC on Windows")
    (root / "build").mkdir(parents=True, exist_ok=True)
    includes = _flags(root, "cl_includes")
    flags = _flags(root, "cl_asan_flags") + includes
    targets = (
        ("test_codec_sources", "build/test_codec_asan.exe", ()),
        ("test_quantizer_sources", "build/test_quantizer_asan.exe", ()),
        ("test_avx2_sources", "build/test_quantizer_avx2_asan.exe", ("/arch:AVX2", "/openmp")),
        ("test_rng_cpu_sources", "build/test_rng_cpu_asan.exe", ()),
        ("test_asan_probe_sources", "build/asan_overread_probe.exe", ()),
    )
    for variable, output, extra in targets:
        command_flags = (
            flags if variable != "test_asan_probe_sources" else _flags(root, "cl_asan_flags")
        )
        _run_command(
            root,
            [
                "cl.exe",
                *command_flags,
                *extra,
                *just_var(root, variable),
                "/Fo:build/",
                f"/Fe:{output}",
            ],
        )
    wrapper = str(root / "tools" / "with-msvc.ps1")
    for binary in (
        "build/test_codec_asan.exe",
        "build/test_quantizer_asan.exe",
        "build/test_quantizer_avx2_asan.exe",
        "build/test_rng_cpu_asan.exe",
    ):
        _run_command(
            root,
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                wrapper,
                f"./{binary}",
            ],
        )
    _run_command(
        root,
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            wrapper,
            "./tools/check_asan_overread.ps1",
            "build/asan_overread_probe.exe",
        ],
    )


def build(root: Path, recipe: str) -> Path:
    if recipe == "test-cpu-gcc-ubsan":
        _run_test_suite(root)
        return root / "build" / "stoquant_gcc_ubsan"
    if recipe == "test-cpu-asan":
        _run_asan_suite(root)
        return root / "build"
    commands, output = command_plan(root, recipe)
    if recipe == "build-cuda-fault":
        commands, output = _cuda_fault_plan(root)
    (root / "build").mkdir(parents=True, exist_ok=True)
    for directory in ("cpu", "cuda", "test-avx2"):
        (root / "build" / directory).mkdir(parents=True, exist_ok=True)
    for command in commands:
        _run_command(root, command)
    binary = _path(root, output)
    from stoquant.build_stamp import write_stamp

    write_stamp(root, binary, recipe)
    return binary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    args = parser.parse_args(argv)
    try:
        build(Path.cwd(), args.recipe)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"native build failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
