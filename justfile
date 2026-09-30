set windows-shell := ["pwsh", "-NoProfile", "-Command"]

default: verify

# Display the current repository state
status:
    git status --short --branch

# Format Python and C/CUDA sources and apply Ruff's safe lint fixes
format:
    uv run ruff check --fix .
    uv run ruff format .
    uv run python tools/clang_format.py

# Verify formatting without changing files
format-check:
    uv run ruff format --check .
    uv run python tools/clang_format.py --check

# Lint Python (Ruff, basedpyright) and the native sources
lint: lint-python lint-native

lint-python:
    uv run ruff check .
    uv run basedpyright

# clang-tidy and MSVC /analyze on every translation unit the build compiles
[windows]
lint-native:
    ./tools/with-msvc.ps1 uv run python tools/native_lint.py

[unix]
lint-native:
    @echo "native lint needs the MSVC toolchain; run it on Windows"

# Run the non-mutating code-quality gates
verify: format-check lint

# CUDA target architecture; RTX 5060 is sm_120. Override with CUDA_ARCH.
cuda_arch := env("CUDA_ARCH", "native")
# Each compiler flag set is defined once. Includes stay separate so the CUDA
# host build can put -DSQ_ENABLE_CUDA ahead of them.
cl_includes := "/Inative /Ithird_party/random123/include"
cc_includes := "-Inative -Ithird_party/random123/include"
cl_strict_flags := "/W4 /WX /std:c11 /fp:strict /D_CRT_SECURE_NO_WARNINGS"
cl_test_flags := "/nologo " + cl_strict_flags
cl_asan_flags := cl_test_flags + " /fsanitize=address /Zi"
cl_host_flags := "/nologo /O2 " + cl_strict_flags
cc_strict_flags := "-std=c11 -Wall -Wextra -Werror -ffp-contract=off"
cc_host_flags := "-O2 " + cc_strict_flags
nvcc_flags := "-O2 -arch=" + cuda_arch + " " + cc_includes
# Host warnings are errors through nvcc too. C4068 is the unknown-pragma warning
# for the nvcc diag_suppress pragmas in Random123/array.h.
nvcc_warn_flags := if os() == "windows" { "--Werror all-warnings -Xcompiler /W4 -Xcompiler /WX -Xcompiler /wd4068" } else { "--Werror all-warnings" }
nvcc_fp_flags := "--fmad=false --ftz=false --prec-div=true --prec-sqrt=true"
native_host_sources := "native/main.c native/cli.c native/bench_report.c native/sq_status.c native/codec.c native/quantizer.c native/rng_cpu.c"
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
# Only the AVX2 comparator gets vector, OpenMP and non-strict FP flags. Neither
# /fp:precise nor -ffp-contract=off contracts a*b+c, and its tests pin the bytes.
# The file reads float storage as uint32_t, so gcc drops type-based aliasing.
avx2_cl_flags := "/nologo /O2 /W4 /WX /std:c11 /fp:precise /arch:AVX2 /openmp /D_CRT_SECURE_NO_WARNINGS " + cl_includes
avx2_cc_flags := "-O2 -std=c11 -Wall -Wextra -Werror -mavx2 -fopenmp -ffp-contract=off -fno-strict-aliasing " + cc_includes

# Record the CUDA toolkit and GPU used for a build
[windows]
toolchain:
    nvcc --version
    ./tools/with-msvc.ps1 cl.exe 2>&1 | Select-Object -First 1
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

[unix]
toolchain:
    nvcc --version
    ${CXX:-g++} --version | head -n 1
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

# Build the Philox known-answer and mapping test executable
[windows]
build-rng:
    uv run python -m stoquant.native_build build-rng

[unix]
build-rng:
    uv run python -m stoquant.native_build build-rng

# Build the Philox known-answer, mapping and overflow checks without CUDA
[windows]
build-rng-cpu-test:
    uv run python -m stoquant.native_build build-rng-cpu-test

[unix]
build-rng-cpu-test:
    uv run python -m stoquant.native_build build-rng-cpu-test

# Run the RNG checks on CPU and GPU; exits nonzero on any failure
test-rng: build-rng
    ./build/test_rng

# Build the CPU compression and decompression command-line tool
[windows]
build-cpu:
    uv run python -m stoquant.native_build build-cpu

[unix]
build-cpu:
    uv run python -m stoquant.native_build build-cpu

# Build the CUDA-enabled 4-bit and 8-bit compression CLI. Keep host sources in C mode
# so the CPU and CUDA paths share the same C implementation and ABI.
[windows]
build-cuda:
    uv run python -m stoquant.native_build build-cuda

[unix]
build-cuda:
    uv run python -m stoquant.native_build build-cuda

# Build a fault-injection CLI beside the evidence binary (test only; never
# overwrites build/stoquant)
[windows]
build-cuda-fault: build-cuda
    uv run python -m stoquant.native_build build-cuda-fault

[unix]
build-cuda-fault: build-cuda
    uv run python -m stoquant.native_build build-cuda-fault

# Build the device-attribute and streaming-read probe for the K1 baseline
[windows]
build-stream-probe:
    uv run python -m stoquant.native_build build-stream-probe

[unix]
build-stream-probe:
    uv run python -m stoquant.native_build build-stream-probe

[windows]
build-codec-test:
    uv run python -m stoquant.native_build build-codec-test

[unix]
build-codec-test:
    uv run python -m stoquant.native_build build-codec-test

# Build the AVX2-versus-scalar differential test
[windows]
build-avx2-test:
    uv run python -m stoquant.native_build build-avx2-test

[unix]
build-avx2-test:
    uv run python -m stoquant.native_build build-avx2-test

# Check that MSVC reports every tagged AVX2 hot loop as vectorized
[windows]
vec-report:
    uv run python -m stoquant vec-report

[windows]
build-quantizer-test:
    uv run python -m stoquant.native_build build-quantizer-test

[unix]
build-quantizer-test:
    uv run python -m stoquant.native_build build-quantizer-test

# Build the CPU tool, verify the C seams, and run the independent Python oracle/CLI suite
[windows]
test-cpu: build-cpu build-codec-test build-quantizer-test build-avx2-test build-rng-cpu-test
    ./build/test_codec.exe
    ./build/test_quantizer.exe
    ./build/test_quantizer_avx2.exe
    ./build/test_rng_cpu.exe
    $env:STOQUANT_EXPECT_CPU_ONLY = '1'; uv run --group dev pytest

[unix]
test-cpu: build-cpu build-codec-test build-quantizer-test build-avx2-test build-rng-cpu-test
    ./build/test_codec
    ./build/test_quantizer
    ./build/test_quantizer_avx2
    ./build/test_rng_cpu
    STOQUANT_EXPECT_CPU_ONLY=1 uv run --group dev pytest

# Build and run every native CPU test executable under AddressSanitizer
[windows]
test-cpu-asan:
    uv run python -m stoquant.native_build test-cpu-asan

[unix]
test-cpu-asan:
    @echo "the AddressSanitizer CPU proof uses MSVC on Windows"

# Compile the command-line tool and run all native C tests under GCC UBSan
[unix]
test-cpu-gcc-ubsan:
    uv run python -m stoquant.native_build test-cpu-gcc-ubsan

# Run the CUDA quantizer acceptance suite on the local GPU
[windows]
test-cuda: build-cuda build-cuda-fault
    $env:STOQUANT_TEST_CUDA = '1'; uv run --group dev pytest --require-cuda tests\test_cuda.py tests\test_matrix_driver.py tests\test_sweep_avx2.py

[unix]
test-cuda: build-cuda build-cuda-fault
    STOQUANT_TEST_CUDA=1 uv run --group dev pytest --require-cuda tests/test_cuda.py tests/test_matrix_driver.py tests/test_sweep_avx2.py

# Render F1-F4, T1 and the crossover report into a snapshot folder
figures snapshot *args:
    uv run python -m stoquant figures "{{snapshot}}" {{args}}

# Build the CUDA-enabled tool and run the full course benchmark matrix
[windows]
bench-matrix *args: build-cuda build-stream-probe
    uv run python -m stoquant.notify -- python -m stoquant bench-matrix {{args}}

[unix]
bench-matrix *args: build-cuda build-stream-probe
    uv run python -m stoquant.notify -- python -m stoquant bench-matrix {{args}}

# Build the CUDA-enabled tool and run the Layer 3 expectation suite and its figure
[windows]
unbiasedness *args: build-cuda
    uv run python -m stoquant unbiasedness {{args}}

[unix]
unbiasedness *args: build-cuda
    uv run python -m stoquant unbiasedness {{args}}

# Measure the reference K1 against the device bandwidth (issue #23 baseline)
[windows]
k1-baseline *args: build-cuda build-stream-probe
    uv run python -m stoquant k1-baseline {{args}}

[unix]
k1-baseline *args: build-cuda build-stream-probe
    uv run python -m stoquant k1-baseline {{args}}

# A/B the reference and optimized K1 on the same tree, or re-render F5 (issue #23)
[windows]
k1-ab *args: build-cuda build-stream-probe
    uv run python -m stoquant k1-ab {{args}}

[unix]
k1-ab *args: build-cuda build-stream-probe
    uv run python -m stoquant k1-ab {{args}}

# Capture or compare the restructure equivalence baseline (issue #47)
equivalence *args: build-cuda build-stream-probe
    uv run python tools/equivalence.py {{args}}

# Fast, bounded record-parity smoke for routine implementation changes
equivalence-smoke:
    uv run python tools/equivalence_smoke.py
