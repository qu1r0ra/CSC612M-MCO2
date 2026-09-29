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
cl_host_flags := "/nologo /O2 " + cl_strict_flags
cc_strict_flags := "-std=c11 -Wall -Wextra -Werror -ffp-contract=off"
cc_host_flags := "-O2 " + cc_strict_flags
nvcc_flags := "-O2 -arch=" + cuda_arch + " " + cc_includes
# Host warnings are errors through nvcc too. C4068 is the unknown-pragma warning
# for the nvcc diag_suppress pragmas in Random123/array.h.
nvcc_warn_flags := if os() == "windows" { "--Werror all-warnings -Xcompiler /W4 -Xcompiler /WX -Xcompiler /wd4068" } else { "--Werror all-warnings" }
nvcc_fp_flags := "--fmad=false --ftz=false --prec-div=true --prec-sqrt=true"
rng_sources := "native/rng_cpu.c native/rng_cuda.cu tests/test_rng.c"
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
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 nvcc {{nvcc_flags}} {{nvcc_warn_flags}} {{rng_sources}} -o build/test_rng.exe

[unix]
build-rng:
    mkdir -p build
    nvcc {{nvcc_flags}} {{nvcc_warn_flags}} {{rng_sources}} -o build/test_rng

# Run the RNG checks on CPU and GPU; exits nonzero on any failure
test-rng: build-rng
    ./build/test_rng

# Build the CPU compression and decompression command-line tool
[windows]
build-cpu:
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\main.c /Fo:build\main.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\cli.c /Fo:build\cli.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\bench_report.c /Fo:build\bench_report.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\codec.c /Fo:build\codec.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\sq_status.c /Fo:build\sq_status.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\quantizer.c /Fo:build\quantizer.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} /c native\rng_cpu.c /Fo:build\rng_cpu.obj
    ./tools/with-msvc.ps1 cl.exe {{avx2_cl_flags}} /c native\quantizer_avx2.c /Fo:build\quantizer_avx2.obj
    ./tools/with-msvc.ps1 cl.exe /nologo build\main.obj build\cli.obj build\bench_report.obj build\sq_status.obj build\codec.obj build\quantizer.obj build\rng_cpu.obj build\quantizer_avx2.obj /Fe:build\stoquant.exe

[unix]
build-cpu:
    mkdir -p build
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/main.c -o build/main.o
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/cli.c -o build/cli.o
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/bench_report.c -o build/bench_report.o
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/codec.c -o build/codec.o
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/sq_status.c -o build/sq_status.o
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/quantizer.c -o build/quantizer.o
    ${CC:-cc} {{cc_host_flags}} {{cc_includes}} -c native/rng_cpu.c -o build/rng_cpu.o
    ${CC:-cc} {{avx2_cc_flags}} -c native/quantizer_avx2.c -o build/quantizer_avx2.o
    ${CC:-cc} -fopenmp build/main.o build/cli.o build/bench_report.o build/sq_status.o build/codec.o build/quantizer.o build/rng_cpu.o build/quantizer_avx2.o -lm -o build/stoquant

# Build the CUDA-enabled 4-bit and 8-bit compression CLI. Keep host sources in C mode
# so the CPU and CUDA paths share the same C implementation and ABI.
[windows]
build-cuda:
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\main.c /Fo:build\main_cuda.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\cli.c /Fo:build\cli_cuda.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\bench_report.c /Fo:build\bench_report_cuda.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\codec.c /Fo:build\codec_cuda.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\sq_status.c /Fo:build\sq_status_cuda.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\quantizer.c /Fo:build\quantizer_cuda_host.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} /DSQ_ENABLE_CUDA {{cl_includes}} /c native\rng_cpu.c /Fo:build\rng_cpu_cuda.obj
    ./tools/with-msvc.ps1 cl.exe {{avx2_cl_flags}} /c native\quantizer_avx2.c /Fo:build\quantizer_avx2_cuda.obj
    ./tools/with-msvc.ps1 nvcc {{nvcc_flags}} {{nvcc_fp_flags}} {{nvcc_warn_flags}} -c native\quantizer_cuda.cu -o build\quantizer_cuda.obj
    ./tools/with-msvc.ps1 nvcc {{nvcc_flags}} build\main_cuda.obj build\cli_cuda.obj build\bench_report_cuda.obj build\sq_status_cuda.obj build\codec_cuda.obj build\quantizer_cuda_host.obj build\rng_cpu_cuda.obj build\quantizer_avx2_cuda.obj build\quantizer_cuda.obj -o build\stoquant.exe

[unix]
build-cuda:
    mkdir -p build
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/main.c -o build/main_cuda.o
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/cli.c -o build/cli_cuda.o
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/bench_report.c -o build/bench_report_cuda.o
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/codec.c -o build/codec_cuda.o
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/sq_status.c -o build/sq_status_cuda.o
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/quantizer.c -o build/quantizer_cuda_host.o
    ${CC:-cc} {{cc_host_flags}} -DSQ_ENABLE_CUDA {{cc_includes}} -c native/rng_cpu.c -o build/rng_cpu_cuda.o
    ${CC:-cc} {{avx2_cc_flags}} -c native/quantizer_avx2.c -o build/quantizer_avx2_cuda.o
    nvcc {{nvcc_flags}} {{nvcc_fp_flags}} {{nvcc_warn_flags}} -c native/quantizer_cuda.cu -o build/quantizer_cuda.o
    nvcc {{nvcc_flags}} {{nvcc_fp_flags}} build/main_cuda.o build/cli_cuda.o build/bench_report_cuda.o build/sq_status_cuda.o build/codec_cuda.o build/quantizer_cuda_host.o build/rng_cpu_cuda.o build/quantizer_avx2_cuda.o build/quantizer_cuda.o -lm -Xcompiler -fopenmp -o build/stoquant

# Build a fault-injection CLI beside the evidence binary (test only; never
# overwrites build/stoquant)
[windows]
build-cuda-fault: build-cuda
    ./tools/with-msvc.ps1 nvcc {{nvcc_flags}} {{nvcc_fp_flags}} {{nvcc_warn_flags}} -DSQ_CUDA_FAULT_INJECTION -c native\quantizer_cuda.cu -o build\quantizer_cuda_fault.obj
    ./tools/with-msvc.ps1 nvcc {{nvcc_flags}} build\main_cuda.obj build\cli_cuda.obj build\bench_report_cuda.obj build\sq_status_cuda.obj build\codec_cuda.obj build\quantizer_cuda_host.obj build\rng_cpu_cuda.obj build\quantizer_avx2_cuda.obj build\quantizer_cuda_fault.obj -o build\stoquant_fault.exe

[unix]
build-cuda-fault: build-cuda
    nvcc {{nvcc_flags}} {{nvcc_fp_flags}} {{nvcc_warn_flags}} -DSQ_CUDA_FAULT_INJECTION -c native/quantizer_cuda.cu -o build/quantizer_cuda_fault.o
    nvcc {{nvcc_flags}} {{nvcc_fp_flags}} build/main_cuda.o build/cli_cuda.o build/bench_report_cuda.o build/sq_status_cuda.o build/codec_cuda.o build/quantizer_cuda_host.o build/rng_cpu_cuda.o build/quantizer_avx2_cuda.o build/quantizer_cuda_fault.o -lm -Xcompiler -fopenmp -o build/stoquant_fault

# Build the device-attribute and streaming-read probe for the K1 baseline
[windows]
build-stream-probe:
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 nvcc {{nvcc_flags}} {{nvcc_warn_flags}} native\stream_probe.cu -o build\stream_probe.exe

[unix]
build-stream-probe:
    mkdir -p build
    nvcc {{nvcc_flags}} {{nvcc_warn_flags}} native/stream_probe.cu -o build/stream_probe

[windows]
build-codec-test:
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 cl.exe {{cl_test_flags}} {{cl_includes}} tests\test_codec.c native\codec.c /Fo:build\ /Fe:build\test_codec.exe

[unix]
build-codec-test:
    mkdir -p build
    ${CC:-cc} {{cc_strict_flags}} {{cc_includes}} tests/test_codec.c native/codec.c -o build/test_codec

# Build the AVX2-versus-scalar differential test
[windows]
build-avx2-test:
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 cl.exe {{cl_test_flags}} {{cl_includes}} /c tests\test_quantizer_avx2.c /Fo:build\test_quantizer_avx2.obj
    ./tools/with-msvc.ps1 cl.exe {{avx2_cl_flags}} /c native\quantizer_avx2.c /Fo:build\quantizer_avx2_test.obj
    ./tools/with-msvc.ps1 cl.exe {{cl_host_flags}} {{cl_includes}} build\test_quantizer_avx2.obj build\quantizer_avx2_test.obj native\quantizer.c native\codec.c native\rng_cpu.c /Fo:build\ /Fe:build\test_quantizer_avx2.exe

[unix]
build-avx2-test:
    mkdir -p build
    ${CC:-cc} {{cc_strict_flags}} {{cc_includes}} -c tests/test_quantizer_avx2.c -o build/test_quantizer_avx2.o
    ${CC:-cc} {{avx2_cc_flags}} -c native/quantizer_avx2.c -o build/quantizer_avx2_test.o
    ${CC:-cc} {{cc_host_flags}} -fopenmp {{cc_includes}} build/test_quantizer_avx2.o build/quantizer_avx2_test.o native/quantizer.c native/codec.c native/rng_cpu.c -lm -o build/test_quantizer_avx2

# Check that MSVC reports every tagged AVX2 hot loop as vectorized
[windows]
vec-report:
    uv run python -m stoquant vec-report

[windows]
build-quantizer-test:
    New-Item -ItemType Directory -Force build | Out-Null
    ./tools/with-msvc.ps1 cl.exe {{cl_test_flags}} {{cl_includes}} tests\test_quantizer.c native\quantizer.c native\codec.c native\rng_cpu.c /Fo:build\ /Fe:build\test_quantizer.exe

[unix]
build-quantizer-test:
    mkdir -p build
    ${CC:-cc} {{cc_strict_flags}} {{cc_includes}} tests/test_quantizer.c native/quantizer.c native/codec.c native/rng_cpu.c -lm -o build/test_quantizer

# Build the CPU tool, verify the C seams, and run the independent Python oracle/CLI suite
[windows]
test-cpu: build-cpu build-codec-test build-quantizer-test build-avx2-test
    ./build/test_codec.exe
    ./build/test_quantizer.exe
    ./build/test_quantizer_avx2.exe
    $env:STOQUANT_EXPECT_CPU_ONLY = '1'; uv run --group dev pytest

[unix]
test-cpu: build-cpu build-codec-test build-quantizer-test build-avx2-test
    ./build/test_codec
    ./build/test_quantizer
    ./build/test_quantizer_avx2
    STOQUANT_EXPECT_CPU_ONLY=1 uv run --group dev pytest

# Run the CUDA quantizer acceptance suite on the local GPU
[windows]
test-cuda: build-cuda build-cuda-fault
    $env:STOQUANT_TEST_CUDA = '1'; uv run --group dev pytest --require-cuda tests\test_cuda.py tests\test_bench_driver.py

[unix]
test-cuda: build-cuda build-cuda-fault
    STOQUANT_TEST_CUDA=1 uv run --group dev pytest --require-cuda tests/test_cuda.py tests/test_bench_driver.py

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
