set windows-shell := ["pwsh", "-NoProfile", "-Command"]

default: verify

# Display the current repository state
status:
    git status --short --branch

# Apply Ruff's formatter to Python files
format:
    ruff format .

# Run Ruff's linter
lint:
    ruff check .

# Verify that Ruff formatting is already clean without changing files
format-check:
    ruff format --check .

# Run the non-mutating code-quality gates
verify: lint format-check

# CUDA target architecture; RTX 5060 is sm_120. Override with CUDA_ARCH.
cuda_arch := env("CUDA_ARCH", "native")
nvcc_flags := "-O2 -arch=" + cuda_arch + " -Isrc -Ithird_party/random123/include"
rng_sources := "src/rng_cpu.c src/rng_cuda.cu tests/test_rng.c"

# Record the CUDA toolkit and GPU used for a build
[windows]
toolchain:
    nvcc --version
    ./scripts/with-msvc.ps1 cl.exe 2>&1 | Select-Object -First 1
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
    ./scripts/with-msvc.ps1 nvcc {{nvcc_flags}} -Xcompiler /wd4068 {{rng_sources}} -o build/test_rng.exe

[unix]
build-rng:
    mkdir -p build
    nvcc {{nvcc_flags}} {{rng_sources}} -o build/test_rng

# Run the RNG checks on CPU and GPU; exits nonzero on any failure
test-rng: build-rng
    ./build/test_rng
