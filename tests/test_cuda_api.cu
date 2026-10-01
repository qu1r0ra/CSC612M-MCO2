#include "quantizer_cuda.h"

#include <cstdint>
#include <cstdio>

static int failures;

static void check(sq_status actual, sq_status expected, const char *message) {
  if (actual != expected) {
    std::fprintf(stderr, "FAIL: %s (got %d, expected %d)\n", message,
                 static_cast<int>(actual), static_cast<int>(expected));
    failures++;
  }
}

int main() {
  sq_cuda_config config = {};
  sq_bench_sample samples[2] = {};
  float scale = -1.0f;

  config.bit_width = SQ_Q8_BITS;
  config.tensor_id = 1;
  config.invocation_id = UINT32_MAX;
  config.block_size = 128;
  config.grid_size = 1;
  config.boundary = SQ_CUDA_BENCH_RESIDENT;
  config.transfer_policy = SQ_CUDA_TRANSFER_PAGEABLE;
  config.reps = 2;

  config.invocation_id_step = 1;
  check(sq_cuda_bench(&config, nullptr, &scale, samples, nullptr),
        SQ_ERR_ID_OVERFLOW,
        "advancing runs beyond UINT32_MAX fail through the CUDA API");

  config.invocation_id_step = 0;
  config.warmups = UINT64_MAX;
  config.reps = 1;
  check(sq_cuda_bench(&config, nullptr, &scale, samples, nullptr),
        SQ_ERR_ID_OVERFLOW, "warmup plus repetition overflow is rejected");

  config.warmups = 0;
  config.reps = 2;
  config.tensor_id = static_cast<uint64_t>(UINT32_MAX) + 1;
  check(sq_cuda_bench(&config, nullptr, &scale, samples, nullptr),
        SQ_ERR_ID_OVERFLOW,
        "tensor identifiers beyond UINT32_MAX are rejected");

  config.tensor_id = 1;
  check(sq_cuda_bench(&config, nullptr, &scale, samples, nullptr), SQ_OK,
        "fixed replay accepts multiple runs at UINT32_MAX");

  std::printf("# %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
