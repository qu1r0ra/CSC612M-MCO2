#include "quantizer_cuda.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define SQ_CUDA_REDUCTION_THREADS 256

#ifdef SQ_CUDA_FAULT_INJECTION
/* Test build only. STOQUANT_FAULT_CUDA_CALL=n makes the n-th CUDA call of
   sq_cuda_compress fail with cudaErrorUnknown; unset, the gate never fires. */
static int fault_countdown;

static void fault_arm(void) {
  const char *setting = getenv("STOQUANT_FAULT_CUDA_CALL");
  fault_countdown = setting != NULL ? atoi(setting) : 0;
}

static int fault_fires(void) {
  if (fault_countdown > 0 && --fault_countdown == 0) {
    (void)fputs("injected CUDA fault\n", stderr);
    return 1;
  }
  return 0;
}
#define SQ_CUDA_FAULT_ARM() fault_arm()
#define SQ_CUDA_CALL(call) (fault_fires() ? cudaErrorUnknown : (call))
#else
#define SQ_CUDA_FAULT_ARM() ((void)0)
#define SQ_CUDA_CALL(call) (call)
#endif

#define SQ_CUDA_TRY(error, call, failure)                                      \
  do {                                                                         \
    (error) = SQ_CUDA_CALL(call);                                              \
    if ((error) != cudaSuccess) {                                              \
      failure;                                                                 \
    }                                                                          \
  } while (0)

static cudaError_t create_events(cudaEvent_t *events, size_t count) {
  cudaError_t error;
  size_t index;

  for (index = 0; index < count; index++) {
    SQ_CUDA_TRY(error, cudaEventCreate(&events[index]), return error);
  }
  return cudaSuccess;
}

static void destroy_events(cudaEvent_t *events, size_t count) {
  size_t index;

  for (index = 0; index < count; index++) {
    if (events[index] != NULL) {
      (void)cudaEventDestroy(events[index]);
    }
  }
}

static int cuda_device_available(void) {
  int device_count = 0;
  cudaError_t error;

  SQ_CUDA_TRY(error, cudaGetDeviceCount(&device_count), {
    (void)cudaGetLastError();
    return 0;
  });
  if (device_count == 0) {
    (void)cudaGetLastError();
    return 0;
  }
  return 1;
}

enum { SQ_CUDA_TIMING_START, SQ_CUDA_TIMING_STOP, SQ_CUDA_TIMING_COUNT };

enum {
  SQ_CUDA_INPUT_NONFINITE = 1U,
  SQ_CUDA_INPUT_ANY_NONZERO = 2U,
  SQ_CUDA_INPUT_BAD_ZERO_SCALE = 4U
};

__global__ static void max_blocks_kernel(const float *values, uint64_t count,
                                         uint64_t block_count,
                                         float *max_partials,
                                         uint32_t *invalid_partials) {
  __shared__ float block_max[SQ_CUDA_REDUCTION_THREADS];
  __shared__ uint32_t block_invalid[SQ_CUDA_REDUCTION_THREADS];
  const unsigned int lane = threadIdx.x;
  uint64_t logical_block;

  for (logical_block = blockIdx.x; logical_block < block_count;
       logical_block += gridDim.x) {
    const uint64_t index = logical_block * SQ_CUDA_REDUCTION_THREADS + lane;
    float value_max = 0.0f;
    uint32_t invalid = 0;

    if (index < count) {
      const float value = values[index];
      if (isfinite(value)) {
        value_max = fabsf(value);
      } else {
        invalid = 1;
      }
    }
    block_max[lane] = value_max;
    block_invalid[lane] = invalid;
    __syncthreads();

    for (unsigned int stride = SQ_CUDA_REDUCTION_THREADS / 2; stride != 0;
         stride /= 2) {
      if (lane < stride) {
        const float other_max = block_max[lane + stride];
        if (other_max > block_max[lane]) {
          block_max[lane] = other_max;
        }
        block_invalid[lane] |= block_invalid[lane + stride];
      }
      __syncthreads();
    }

    if (lane == 0) {
      max_partials[logical_block] = block_max[0];
      invalid_partials[logical_block] = block_invalid[0];
    }
    __syncthreads();
  }
}

__global__ static void reduce_max_kernel(const float *max_partials,
                                         const uint32_t *invalid_partials,
                                         uint64_t block_count, float *scale,
                                         int *status) {
  __shared__ float maxima[SQ_CUDA_REDUCTION_THREADS];
  __shared__ uint32_t invalid[SQ_CUDA_REDUCTION_THREADS];
  const unsigned int lane = threadIdx.x;
  float local_max = 0.0f;
  uint32_t local_invalid = 0;

  for (uint64_t i = lane; i < block_count; i += SQ_CUDA_REDUCTION_THREADS) {
    if (max_partials[i] > local_max) {
      local_max = max_partials[i];
    }
    local_invalid |= invalid_partials[i];
  }
  maxima[lane] = local_max;
  invalid[lane] = local_invalid;
  __syncthreads();

  for (unsigned int stride = SQ_CUDA_REDUCTION_THREADS / 2; stride != 0;
       stride /= 2) {
    if (lane < stride) {
      if (maxima[lane + stride] > maxima[lane]) {
        maxima[lane] = maxima[lane + stride];
      }
      invalid[lane] |= invalid[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    *scale = maxima[0];
    *status = invalid[0] ? SQ_ERR_NONFINITE : SQ_OK;
  }
}

__global__ static void sum_blocks_kernel(const float *values, uint64_t count,
                                         uint64_t block_count,
                                         uint64_t padded_count,
                                         const float *max_abs,
                                         const int *status, float *partials) {
  __shared__ float terms[2][SQ_CUDA_REDUCTION_THREADS];
  const unsigned int lane = threadIdx.x;
  uint64_t logical_block;

  for (logical_block = blockIdx.x; logical_block < padded_count;
       logical_block += gridDim.x) {
    if (logical_block >= block_count || *status != SQ_OK || *max_abs == 0.0f) {
      if (lane == 0) {
        partials[logical_block] = 0.0f;
      }
      __syncthreads();
      continue;
    }

    const uint64_t index = logical_block * SQ_CUDA_REDUCTION_THREADS + lane;
    float term = 0.0f;
    if (index < count) {
      const float ratio = fabsf(values[index]) / *max_abs;
      term = ratio * ratio;
    }
    float *input = terms[0];
    float *output = terms[1];
    input[lane] = term;
    __syncthreads();

    for (unsigned int stride = SQ_CUDA_REDUCTION_THREADS / 2; stride != 0;
         stride /= 2) {
      if (lane < stride) {
        // 2 * lane < SQ_CUDA_REDUCTION_THREADS cannot overflow, and hoisting
        // the index changes this timed kernel's SASS.
        // NOLINTBEGIN(bugprone-implicit-widening-of-multiplication-result)
        output[lane] = input[2 * lane] + input[2 * lane + 1];
        // NOLINTEND(bugprone-implicit-widening-of-multiplication-result)
      }
      __syncthreads();
      float *temporary = input;
      input = output;
      output = temporary;
    }
    if (lane == 0) {
      partials[logical_block] = input[0];
    }
    __syncthreads();
  }
}

__global__ static void reduce_pairs_kernel(const float *input, float *output,
                                           uint64_t output_count) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < output_count; i += stride) {
    output[i] = input[2 * i] + input[2 * i + 1];
  }
}

__global__ static void finish_scale_kernel(const float *sum, float *scale,
                                           int *status) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || *status != SQ_OK) {
    return;
  }
  if (*scale == 0.0f) {
    return;
  }

  const float root = sqrtf(*sum);
  const float result = *scale * root;
  if (!isfinite(result)) {
    *status = SQ_ERR_SCALE_OVERFLOW;
    *scale = 0.0f;
  } else {
    *scale = result;
  }
}

__global__ static void round_kernel(uint8_t bit_width, const float *values,
                                    uint64_t count, const float *scale,
                                    const uint32_t *words, int prescribed_words,
                                    sq_rng_stream stream_state, uint8_t *codes,
                                    uint32_t *validation_flags) {
  const int s =
      (bit_width == SQ_Q4_BITS) ? SQ_Q4_SIGNED_LIMIT : SQ_Q8_SIGNED_LIMIT;
  const philox4x32_key_t key = sq_philox_key(&stream_state);
  const uint64_t group_count = count / 4 + (count % 4 != 0);
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  const float scale_value = *scale;

  for (uint64_t group = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       group < group_count; group += stride) {
    philox4x32_ctr_t random_words;
    if (!prescribed_words) {
      random_words = philox4x32_R(SQ_PHILOX_ROUNDS,
                                  sq_philox_ctr(&stream_state, group), key);
    }
    for (unsigned int lane = 0; lane < 4; lane++) {
      const uint64_t index = group * 4 + lane;
      if (index >= count) {
        break;
      }

      const float value = values[index];
      if (!isfinite(value)) {
        atomicOr(validation_flags, SQ_CUDA_INPUT_NONFINITE);
        continue;
      }
      if (value != 0.0f) {
        atomicOr(validation_flags, SQ_CUDA_INPUT_ANY_NONZERO);
      }
      if (scale_value == 0.0f) {
        if (value != 0.0f) {
          atomicOr(validation_flags, SQ_CUDA_INPUT_BAD_ZERO_SCALE);
        }
        codes[index] = (uint8_t)s;
        continue;
      }

      const float absolute_value = fabsf(value);
      float scaled = (absolute_value / scale_value) * (float)s;
      if (scaled > (float)s) {
        scaled = (float)s;
      }
      const float lower_float = floorf(scaled);
      const float probability = scaled - lower_float;
      const uint32_t word =
          prescribed_words ? words[index] : random_words.v[lane];
      const int magnitude = (int)lower_float + sq_bernoulli(word, probability);
      const int signed_code =
          signbit(value) && magnitude != 0 ? -magnitude : magnitude;
      codes[index] = (uint8_t)(signed_code + s);
    }
  }
}

__global__ static void pack_q8_kernel(const uint8_t *codes, uint64_t count,
                                      uint8_t *payload) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    payload[index] = codes[index];
  }
}

__global__ static void pack_q4_kernel(const uint8_t *codes, uint64_t count,
                                      uint64_t output_bytes, uint8_t *payload) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t byte_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       byte_index < output_bytes; byte_index += stride) {
    const uint64_t idx0 = byte_index * 2;
    const uint64_t idx1 = idx0 + 1;
    const uint8_t low_nibble = (uint8_t)(codes[idx0] & 0x0F);
    const uint8_t high_nibble =
        (idx1 < count) ? (uint8_t)(codes[idx1] & 0x0F) : (uint8_t)0;
    payload[byte_index] = (uint8_t)(low_nibble | (high_nibble << 4));
  }
}

/*
 * Optimized K1 (issue #23). The max is order-free, so each thread scans float4
 * words grid-stride and each block keeps one partial. The sum must reproduce
 * the reference tree bit for bit: one perfect binary tree of adjacent pairs
 * over padded_count * 256 leaves, zero-padded. A warp builds one 256-leaf
 * subtree: each lane adds its 8 contiguous leaves as a 3-level tree, and
 * shuffles with offsets 1, 2, 4, 8, 16 add the next 5 levels. Each launch above
 * that reduces 2048 inputs per block instead of one level per launch.
 */
#define SQ_CUDA_WARP 32
#define SQ_CUDA_K1_LEAVES_PER_LANE 8
#define SQ_CUDA_K1_TREE_SPAN 2048
#define SQ_CUDA_K1_MAX_AUTO_GRID 1024

/* The optimized kernels size their warp arrays for 256-thread blocks. */
static_assert(SQ_CUDA_REDUCTION_THREADS % SQ_CUDA_WARP == 0 &&
                  SQ_CUDA_REDUCTION_THREADS / SQ_CUDA_WARP *
                          SQ_CUDA_K1_LEAVES_PER_LANE <=
                      SQ_CUDA_K1_TREE_SPAN,
              "optimized K1 geometry");

static int k1_variant = SQ_CUDA_K1_REFERENCE;

extern "C" int sq_cuda_select_k1(int variant) {
  if (variant != SQ_CUDA_K1_REFERENCE && variant != SQ_CUDA_K1_OPTIMIZED) {
    return (int)cudaErrorInvalidValue;
  }
  k1_variant = variant;
  return (int)cudaSuccess;
}

__device__ static void absorb_max(float value, float *local_max,
                                  uint32_t *local_invalid) {
  if (!isfinite(value)) {
    *local_invalid = 1;
    return;
  }
  const float magnitude = fabsf(value);
  if (magnitude > *local_max) {
    *local_max = magnitude;
  }
}

__global__ static void max_vector_kernel(const float *values, uint64_t count,
                                         float *max_partials,
                                         uint32_t *invalid_partials) {
  __shared__ float warp_max[SQ_CUDA_REDUCTION_THREADS / SQ_CUDA_WARP];
  __shared__ uint32_t warp_invalid[SQ_CUDA_REDUCTION_THREADS / SQ_CUDA_WARP];
  const float4 *words = reinterpret_cast<const float4 *>(values);
  const uint64_t word_count = count / 4;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int lane = threadIdx.x % SQ_CUDA_WARP;
  const unsigned int warp = threadIdx.x / SQ_CUDA_WARP;
  float local_max = 0.0f;
  uint32_t local_invalid = 0;

  for (uint64_t i = first; i < word_count; i += stride) {
    const float4 word = words[i];
    absorb_max(word.x, &local_max, &local_invalid);
    absorb_max(word.y, &local_max, &local_invalid);
    absorb_max(word.z, &local_max, &local_invalid);
    absorb_max(word.w, &local_max, &local_invalid);
  }
  if (first < count % 4) {
    absorb_max(values[word_count * 4 + first], &local_max, &local_invalid);
  }

  for (unsigned int offset = SQ_CUDA_WARP / 2; offset != 0; offset /= 2) {
    const float other_max = __shfl_down_sync(0xFFFFFFFFU, local_max, offset);
    if (other_max > local_max) {
      local_max = other_max;
    }
    local_invalid |= __shfl_down_sync(0xFFFFFFFFU, local_invalid, offset);
  }
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_invalid[warp] = local_invalid;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned int w = 1; w < SQ_CUDA_REDUCTION_THREADS / SQ_CUDA_WARP;
         w++) {
      if (warp_max[w] > local_max) {
        local_max = warp_max[w];
      }
      local_invalid |= warp_invalid[w];
    }
    max_partials[blockIdx.x] = local_max;
    invalid_partials[blockIdx.x] = local_invalid;
  }
}

__device__ static float scale_term(float value, float max_abs) {
  const float ratio = fabsf(value) / max_abs;
  return ratio * ratio;
}

__global__ static void sum_warps_kernel(const float *values, uint64_t count,
                                        uint64_t block_count,
                                        uint64_t padded_count,
                                        const float *max_abs, const int *status,
                                        float *partials) {
  const unsigned int warps_per_block = SQ_CUDA_REDUCTION_THREADS / SQ_CUDA_WARP;
  const unsigned int lane = threadIdx.x % SQ_CUDA_WARP;
  const uint64_t warp_stride = (uint64_t)gridDim.x * warps_per_block;
  const int skip = *status != SQ_OK || *max_abs == 0.0f;
  const float max_value = *max_abs;

  for (uint64_t chunk =
           (uint64_t)blockIdx.x * warps_per_block + threadIdx.x / SQ_CUDA_WARP;
       chunk < padded_count; chunk += warp_stride) {
    if (skip || chunk >= block_count) {
      if (lane == 0) {
        partials[chunk] = 0.0f;
      }
      continue;
    }

    const uint64_t base = chunk * SQ_CUDA_REDUCTION_THREADS +
                          (uint64_t)lane * SQ_CUDA_K1_LEAVES_PER_LANE;
    float t[SQ_CUDA_K1_LEAVES_PER_LANE];
    if (base + SQ_CUDA_K1_LEAVES_PER_LANE <= count) {
      const float4 low = *reinterpret_cast<const float4 *>(values + base);
      const float4 high = *reinterpret_cast<const float4 *>(values + base + 4);
      t[0] = scale_term(low.x, max_value);
      t[1] = scale_term(low.y, max_value);
      t[2] = scale_term(low.z, max_value);
      t[3] = scale_term(low.w, max_value);
      t[4] = scale_term(high.x, max_value);
      t[5] = scale_term(high.y, max_value);
      t[6] = scale_term(high.z, max_value);
      t[7] = scale_term(high.w, max_value);
    } else {
      for (unsigned int k = 0; k < SQ_CUDA_K1_LEAVES_PER_LANE; k++) {
        t[k] =
            base + k < count ? scale_term(values[base + k], max_value) : 0.0f;
      }
    }
    float sum =
        ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));

    /* After offset d, a lane that is a multiple of 2d holds the subtree of
       lanes l .. l + 2d - 1; other lanes hold values no one reads. */
    for (unsigned int offset = 1; offset < SQ_CUDA_WARP; offset *= 2) {
      sum = sum + __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    if (lane == 0) {
      partials[chunk] = sum;
    }
  }
}

/* Reduces each run of span adjacent inputs (a power of two, at most 2048) to
   one output as a perfect binary tree of adjacent pairs. */
__global__ static void reduce_tree_kernel(const float *input, float *output,
                                          uint64_t output_count,
                                          unsigned int span) {
  __shared__ float level[2][SQ_CUDA_K1_TREE_SPAN];

  for (uint64_t segment = blockIdx.x; segment < output_count;
       segment += gridDim.x) {
    float *source = level[0];
    float *target = level[1];
    for (unsigned int k = threadIdx.x; k < span; k += blockDim.x) {
      source[k] = input[segment * span + k];
    }
    __syncthreads();
    for (unsigned int width = span / 2; width != 0; width /= 2) {
      for (unsigned int k = threadIdx.x; k < width; k += blockDim.x) {
        // 2 * k < span, the shared-memory length, so it cannot overflow; the
        // original index form keeps this timed kernel's generated code.
        // NOLINTBEGIN(bugprone-implicit-widening-of-multiplication-result)
        target[k] = source[2 * k] + source[2 * k + 1];
        // NOLINTEND(bugprone-implicit-widening-of-multiplication-result)
      }
      __syncthreads();
      float *temporary = source;
      source = target;
      target = temporary;
    }
    if (threadIdx.x == 0) {
      output[segment] = source[0];
    }
    __syncthreads();
  }
}

static int launch_grid(uint64_t work, int block_size, int grid_size) {
  if (block_size <= 0) {
    return 0;
  }
  if (grid_size > 0) {
    return grid_size;
  }
  return sq_cuda_automatic_grid(work, block_size);
}

struct sq_cuda_k1_workspace {
  float *max_partials;
  uint32_t *invalid_partials;
  float *sums_a;
  float *sums_b;
  uint64_t block_count;
  uint64_t padded_count;
};

static int valid_launch_geometry(int block_size, int grid_size) {
  return block_size > 0 && block_size <= SQ_CUDA_MAX_BLOCK_SIZE &&
         grid_size >= 0 && grid_size <= SQ_CUDA_MAX_GRID_SIZE;
}

static sq_status set_k1_workspace_geometry(sq_cuda_k1_workspace *workspace,
                                           uint64_t count) {
  uint64_t block_count = count / SQ_CUDA_REDUCTION_THREADS +
                         (count % SQ_CUDA_REDUCTION_THREADS != 0);
  uint64_t padded_count = 1;

  if (workspace == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  if (count == 0) {
    workspace->block_count = 0;
    workspace->padded_count = 0;
    return SQ_OK;
  }
  while (padded_count < block_count) {
    if (padded_count > std::numeric_limits<uint64_t>::max() / 2) {
      return SQ_ERR_COUNT;
    }
    padded_count <<= 1;
  }
  if (block_count > SIZE_MAX / sizeof(float) ||
      block_count > SIZE_MAX / sizeof(uint32_t) ||
      padded_count > SIZE_MAX / sizeof(float)) {
    return SQ_ERR_COUNT;
  }
  workspace->block_count = block_count;
  workspace->padded_count = padded_count;
  return SQ_OK;
}

static cudaError_t allocate_k1_workspace(sq_cuda_k1_workspace *workspace) {
  cudaError_t error;

  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&workspace->max_partials),
                         workspace->block_count * sizeof(float)),
              return error);
  SQ_CUDA_TRY(
      error,
      cudaMalloc(reinterpret_cast<void **>(&workspace->invalid_partials),
                 workspace->block_count * sizeof(uint32_t)),
      return error);
  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&workspace->sums_a),
                         workspace->padded_count * sizeof(float)),
              return error);
  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&workspace->sums_b),
                         workspace->padded_count * sizeof(float)),
              return error);
  return error;
}

static void destroy_k1_workspace(sq_cuda_k1_workspace *workspace) {
  (void)cudaFree(workspace->max_partials);
  (void)cudaFree(workspace->invalid_partials);
  (void)cudaFree(workspace->sums_a);
  (void)cudaFree(workspace->sums_b);
}

static int launch_k1_optimized(const float *device_values, uint64_t count,
                               const sq_cuda_k1_workspace *workspace,
                               float *device_scale, int *device_status,
                               int grid_size, cudaStream_t stream) {
  const uint64_t block_count = workspace->block_count;
  const uint64_t padded_count = workspace->padded_count;
  const uint64_t word_count = count / 4 + (count % 4 != 0);
  const uint64_t auto_max_grid = (uint64_t)sq_cuda_automatic_grid(
      word_count, SQ_CUDA_REDUCTION_THREADS * 2);
  uint64_t max_grid = grid_size > 0 ? (uint64_t)grid_size
                                    : (auto_max_grid < SQ_CUDA_K1_MAX_AUTO_GRID
                                           ? auto_max_grid
                                           : SQ_CUDA_K1_MAX_AUTO_GRID);
  /* One max partial per block, and the buffer holds block_count of them. */
  if (max_grid > block_count) {
    max_grid = block_count;
  }
  const unsigned int sum_grid =
      grid_size > 0
          ? (unsigned int)grid_size
          : (unsigned int)sq_cuda_automatic_grid(
                padded_count, SQ_CUDA_REDUCTION_THREADS / SQ_CUDA_WARP);

  if ((uintptr_t)device_values % sizeof(float4) != 0) {
    return (int)cudaErrorMisalignedAddress;
  }

  max_vector_kernel<<<(unsigned int)max_grid, SQ_CUDA_REDUCTION_THREADS, 0,
                      stream>>>(device_values, count, workspace->max_partials,
                                workspace->invalid_partials);
  cudaError_t error;
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);

  reduce_max_kernel<<<1, SQ_CUDA_REDUCTION_THREADS, 0, stream>>>(
      workspace->max_partials, workspace->invalid_partials, max_grid,
      device_scale, device_status);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);

  sum_warps_kernel<<<sum_grid, SQ_CUDA_REDUCTION_THREADS, 0, stream>>>(
      device_values, count, block_count, padded_count, device_scale,
      device_status, workspace->sums_a);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);

  uint64_t active = padded_count;
  float *input = workspace->sums_a;
  float *output = workspace->sums_b;
  while (active > 1) {
    const unsigned int span = active < SQ_CUDA_K1_TREE_SPAN
                                  ? (unsigned int)active
                                  : SQ_CUDA_K1_TREE_SPAN;
    const uint64_t output_count = active / span;
    const unsigned int tree_grid =
        grid_size > 0 ? (unsigned int)grid_size
                      : (unsigned int)sq_cuda_automatic_grid(output_count, 1);
    reduce_tree_kernel<<<tree_grid, SQ_CUDA_REDUCTION_THREADS, 0, stream>>>(
        input, output, output_count, span);
    SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);
    active = output_count;
    float *temporary = input;
    input = output;
    output = temporary;
  }

  finish_scale_kernel<<<1, 1, 0, stream>>>(input, device_scale, device_status);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);
  return (int)cudaSuccess;
}

static int launch_k1(const float *device_values, uint64_t count,
                     const sq_cuda_k1_workspace *workspace, float *device_scale,
                     int *device_status, int grid_size, cudaStream_t stream) {
  if (count == 0) {
    return (int)cudaSuccess;
  }
  if (!valid_launch_geometry(SQ_CUDA_REDUCTION_THREADS, grid_size) ||
      workspace == NULL) {
    return (int)cudaErrorInvalidValue;
  }
  const uint64_t expected_block_count =
      count / SQ_CUDA_REDUCTION_THREADS +
      (count % SQ_CUDA_REDUCTION_THREADS != 0);
  uint64_t expected_padded_count = 1;
  while (expected_padded_count < expected_block_count) {
    expected_padded_count <<= 1;
  }
  const uint64_t block_count = workspace->block_count;
  const uint64_t padded_count = workspace->padded_count;
  const unsigned int auto_max_grid =
      (unsigned int)(block_count < SQ_CUDA_MAX_GRID_SIZE
                         ? block_count
                         : SQ_CUDA_MAX_GRID_SIZE);
  const unsigned int auto_sum_grid =
      (unsigned int)(padded_count < SQ_CUDA_MAX_GRID_SIZE
                         ? padded_count
                         : SQ_CUDA_MAX_GRID_SIZE);
  const unsigned int max_grid =
      (grid_size > 0) ? (unsigned int)grid_size : auto_max_grid;
  const unsigned int sum_grid =
      (grid_size > 0) ? (unsigned int)grid_size : auto_sum_grid;

  if (block_count != expected_block_count ||
      padded_count != expected_padded_count || max_grid == 0 || sum_grid == 0 ||
      device_values == NULL || workspace->max_partials == NULL ||
      workspace->invalid_partials == NULL || workspace->sums_a == NULL ||
      workspace->sums_b == NULL || device_scale == NULL ||
      device_status == NULL) {
    return (int)cudaErrorInvalidValue;
  }
  if (k1_variant == SQ_CUDA_K1_OPTIMIZED) {
    return launch_k1_optimized(device_values, count, workspace, device_scale,
                               device_status, grid_size, stream);
  }

  max_blocks_kernel<<<max_grid, SQ_CUDA_REDUCTION_THREADS, 0, stream>>>(
      device_values, count, block_count, workspace->max_partials,
      workspace->invalid_partials);
  cudaError_t error;
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);

  reduce_max_kernel<<<1, SQ_CUDA_REDUCTION_THREADS, 0, stream>>>(
      workspace->max_partials, workspace->invalid_partials, block_count,
      device_scale, device_status);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);

  sum_blocks_kernel<<<sum_grid, SQ_CUDA_REDUCTION_THREADS, 0, stream>>>(
      device_values, count, block_count, padded_count, device_scale,
      device_status, workspace->sums_a);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);

  uint64_t active = padded_count;
  float *input = workspace->sums_a;
  float *output = workspace->sums_b;
  while (active > 1) {
    const uint64_t output_count = active / 2;
    const unsigned int auto_pair_grid =
        (unsigned int)sq_cuda_automatic_grid(output_count, 256);
    const unsigned int pair_grid =
        (grid_size > 0) ? (unsigned int)grid_size : auto_pair_grid;
    reduce_pairs_kernel<<<pair_grid, 256, 0, stream>>>(input, output,
                                                       output_count);
    SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);
    active = output_count;
    float *temporary = input;
    input = output;
    output = temporary;
  }

  finish_scale_kernel<<<1, 1, 0, stream>>>(input, device_scale, device_status);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);
  return (int)cudaSuccess;
}

extern "C" int
sq_cuda_launch_k2(uint8_t bit_width, const float *device_values, uint64_t count,
                  const float *device_scale, const uint32_t *device_words,
                  int prescribed_words, sq_rng_stream stream_state,
                  uint8_t *device_codes, uint32_t *device_validation_flags,
                  int block_size, int grid_size, void *stream_handle) {
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return (int)cudaErrorInvalidValue;
  }
  if (count == 0) {
    return 0;
  }
  if (!valid_launch_geometry(block_size, grid_size)) {
    return (int)cudaErrorInvalidValue;
  }
  const uint64_t group_count = count / 4 + (count % 4 != 0);
  const int grid = launch_grid(group_count, block_size, grid_size);
  if (device_values == NULL || device_scale == NULL || device_codes == NULL ||
      device_validation_flags == NULL || grid <= 0 ||
      (prescribed_words && device_words == NULL)) {
    return (int)cudaErrorInvalidValue;
  }

  cudaError_t error;
  round_kernel<<<grid, block_size, 0,
                 reinterpret_cast<cudaStream_t>(stream_handle)>>>(
      bit_width, device_values, count, device_scale, device_words,
      prescribed_words, stream_state, device_codes, device_validation_flags);
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);
  return (int)cudaSuccess;
}

extern "C" int sq_cuda_launch_k3(uint8_t bit_width, const uint8_t *device_codes,
                                 uint64_t count, uint8_t *device_payload,
                                 int block_size, int grid_size,
                                 void *stream_handle) {
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return (int)cudaErrorInvalidValue;
  }
  if (count == 0) {
    return 0;
  }
  if (!valid_launch_geometry(block_size, grid_size)) {
    return (int)cudaErrorInvalidValue;
  }
  const uint64_t output_bytes = sq_payload_size(bit_width, count);
  const int grid = launch_grid(output_bytes, block_size, grid_size);
  if (device_codes == NULL || device_payload == NULL || grid <= 0) {
    return (int)cudaErrorInvalidValue;
  }

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  cudaError_t error;
  if (bit_width == SQ_Q4_BITS) {
    pack_q4_kernel<<<grid, block_size, 0, stream>>>(
        device_codes, count, output_bytes, device_payload);
  } else {
    pack_q8_kernel<<<grid, block_size, 0, stream>>>(device_codes, count,
                                                    device_payload);
  }
  SQ_CUDA_TRY(error, cudaGetLastError(), return (int)error);
  return (int)cudaSuccess;
}

static sq_status timed_copy(void *destination, const void *source, size_t size,
                            cudaMemcpyKind kind, cudaStream_t stream,
                            float *elapsed_ms, int collect_timing) {
  const auto start = std::chrono::steady_clock::now();
  cudaError_t error;

  SQ_CUDA_TRY(error, cudaMemcpyAsync(destination, source, size, kind, stream),
              return SQ_ERR_CUDA);
  SQ_CUDA_TRY(error, cudaStreamSynchronize(stream), return SQ_ERR_CUDA);
  if (collect_timing) {
    const auto end = std::chrono::steady_clock::now();
    *elapsed_ms +=
        std::chrono::duration<float, std::milli>(end - start).count();
  }
  return SQ_OK;
}

static sq_status finish_kernel_timing(cudaEvent_t start, cudaEvent_t stop,
                                      cudaStream_t stream, float *elapsed_ms) {
  cudaError_t error;

  SQ_CUDA_TRY(error, cudaEventRecord(stop, stream), return SQ_ERR_CUDA);
  SQ_CUDA_TRY(error, cudaEventSynchronize(stop), return SQ_ERR_CUDA);
  SQ_CUDA_TRY(error, cudaEventElapsedTime(elapsed_ms, start, stop),
              return SQ_ERR_CUDA);
  return SQ_OK;
}

sq_status sq_cuda_compress(uint8_t bit_width, const float *values, size_t count,
                           uint64_t seed, uint64_t tensor_id,
                           uint64_t invocation_id, int prescribed_scale_seen,
                           float prescribed_scale,
                           const uint32_t *prescribed_words, int block_size,
                           int grid_size, uint8_t *payload, float *scale,
                           int collect_timings, sq_cuda_timings *timings) {
  cudaStream_t stream = NULL;
  cudaEvent_t timing_events[SQ_CUDA_TIMING_COUNT] = {};
  float *device_values = NULL, *device_scale = NULL;
  uint32_t *device_words = NULL;
  uint32_t *device_validation_flags = NULL;
  uint8_t *device_codes = NULL, *device_payload = NULL;
  int *device_status = NULL;
  sq_cuda_k1_workspace k1_workspace = {};
  uint32_t host_validation_flags = 0;
  int host_status = SQ_OK;
  sq_rng_stream stream_state;
  cudaError_t error = cudaSuccess;
  sq_status result = SQ_ERR_CUDA;
  sq_cuda_timings zero_timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const size_t payload_bytes = sq_payload_size(bit_width, count);

  if (timings != NULL) {
    *timings = zero_timings;
  }
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  if (scale == NULL || (count != 0 && (values == NULL || payload == NULL)) ||
      count > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(uint32_t) ||
      (collect_timings && timings == NULL)) {
    return SQ_ERR_ARGUMENT;
  }
  if (!valid_launch_geometry(block_size, grid_size)) {
    return SQ_ERR_ARGUMENT;
  }
  if (prescribed_scale_seen &&
      (!isfinite(prescribed_scale) || prescribed_scale < 0.0f)) {
    return SQ_ERR_SCALE;
  }
  if (sq_rng_stream_init(&stream_state, seed, tensor_id, invocation_id) !=
      SQ_RNG_OK) {
    return SQ_ERR_ID_OVERFLOW;
  }

  SQ_CUDA_FAULT_ARM();
  if (!cuda_device_available()) {
    return SQ_ERR_CUDA;
  }
  if (count == 0 && prescribed_scale_seen && prescribed_scale != 0.0f) {
    return SQ_ERR_SCALE;
  }

  if (count == 0) {
    *scale = prescribed_scale_seen ? prescribed_scale : 0.0f;
    return SQ_OK;
  }

  if (!prescribed_scale_seen) {
    result = set_k1_workspace_geometry(&k1_workspace, count);
    if (result != SQ_OK) {
      return result;
    }
  }

  SQ_CUDA_TRY(error, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              goto done);
  if (collect_timings) {
    error = create_events(timing_events, SQ_CUDA_TIMING_COUNT);
    if (error != cudaSuccess) {
      goto done;
    }
  }

  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&device_values),
                         count * sizeof(float)),
              goto done);
  SQ_CUDA_TRY(
      error,
      cudaMalloc(reinterpret_cast<void **>(&device_scale), sizeof(float)),
      goto done);
  SQ_CUDA_TRY(
      error, cudaMalloc(reinterpret_cast<void **>(&device_status), sizeof(int)),
      goto done);
  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&device_validation_flags),
                         sizeof(uint32_t)),
              goto done);
  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&device_codes),
                         count * sizeof(uint8_t)),
              goto done);
  SQ_CUDA_TRY(
      error,
      cudaMalloc(reinterpret_cast<void **>(&device_payload),
                 (payload_bytes > 0 ? payload_bytes : 1) * sizeof(uint8_t)),
      goto done);
  if (prescribed_words != NULL) {
    SQ_CUDA_TRY(error,
                cudaMalloc(reinterpret_cast<void **>(&device_words),
                           count * sizeof(uint32_t)),
                goto done);
  }
  if (!prescribed_scale_seen) {
    error = allocate_k1_workspace(&k1_workspace);
    if (error != cudaSuccess) {
      goto done;
    }
  }

  SQ_CUDA_TRY(error, cudaMemsetAsync(device_status, 0, sizeof(int), stream),
              goto done);
  SQ_CUDA_TRY(
      error,
      cudaMemsetAsync(device_validation_flags, 0, sizeof(uint32_t), stream),
      goto done);

  result = timed_copy(device_values, values, count * sizeof(float),
                      cudaMemcpyHostToDevice, stream,
                      timings != NULL ? &timings->h2d_ms : NULL,
                      collect_timings && timings != NULL);
  if (result != SQ_OK) {
    goto done;
  }
  if (prescribed_words != NULL) {
    result = timed_copy(device_words, prescribed_words,
                        count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        stream, timings != NULL ? &timings->h2d_ms : NULL,
                        collect_timings && timings != NULL);
    if (result != SQ_OK) {
      goto done;
    }
  }

  if (prescribed_scale_seen) {
    *scale = prescribed_scale;
    result =
        timed_copy(device_scale, scale, sizeof(float), cudaMemcpyHostToDevice,
                   stream, timings != NULL ? &timings->h2d_ms : NULL,
                   collect_timings && timings != NULL);
    if (result != SQ_OK) {
      goto done;
    }
  } else {
    if (collect_timings) {
      SQ_CUDA_TRY(error,
                  cudaEventRecord(timing_events[SQ_CUDA_TIMING_START], stream),
                  goto done);
    }
    SQ_CUDA_TRY(error,
                (cudaError_t)launch_k1(device_values, count, &k1_workspace,
                                       device_scale, device_status, grid_size,
                                       stream),
                goto done);
    if (collect_timings) {
      result = finish_kernel_timing(timing_events[SQ_CUDA_TIMING_START],
                                    timing_events[SQ_CUDA_TIMING_STOP], stream,
                                    &timings->k1_ms);
      if (result != SQ_OK) {
        goto done;
      }
    }
    SQ_CUDA_TRY(error, cudaStreamSynchronize(stream), goto done);
    result = timed_copy(&host_status, device_status, sizeof(int),
                        cudaMemcpyDeviceToHost, stream,
                        timings != NULL ? &timings->d2h_ms : NULL,
                        collect_timings && timings != NULL);
    if (result != SQ_OK) {
      goto done;
    }
    result =
        timed_copy(scale, device_scale, sizeof(float), cudaMemcpyDeviceToHost,
                   stream, timings != NULL ? &timings->d2h_ms : NULL,
                   collect_timings && timings != NULL);
    if (result != SQ_OK) {
      goto done;
    }
    if (host_status != SQ_OK) {
      result = (sq_status)host_status;
      goto done;
    }
  }

  if (collect_timings) {
    SQ_CUDA_TRY(error,
                cudaEventRecord(timing_events[SQ_CUDA_TIMING_START], stream),
                goto done);
  }
  SQ_CUDA_TRY(error,
              (cudaError_t)sq_cuda_launch_k2(
                  bit_width, device_values, count, device_scale, device_words,
                  prescribed_words != NULL, stream_state, device_codes,
                  device_validation_flags, block_size, grid_size, stream),
              goto done);
  if (collect_timings) {
    result = finish_kernel_timing(timing_events[SQ_CUDA_TIMING_START],
                                  timing_events[SQ_CUDA_TIMING_STOP], stream,
                                  &timings->k2_ms);
    if (result != SQ_OK) {
      goto done;
    }
  }
  SQ_CUDA_TRY(error, cudaStreamSynchronize(stream), goto done);
  result = timed_copy(&host_validation_flags, device_validation_flags,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, stream,
                      timings != NULL ? &timings->d2h_ms : NULL,
                      collect_timings && timings != NULL);
  if (result != SQ_OK) {
    goto done;
  }
  if ((host_validation_flags & SQ_CUDA_INPUT_NONFINITE) != 0) {
    result = SQ_ERR_NONFINITE;
    goto done;
  }
  if (((*scale == 0.0f) &&
       (host_validation_flags & SQ_CUDA_INPUT_ANY_NONZERO) != 0) ||
      ((*scale != 0.0f) &&
       (host_validation_flags & SQ_CUDA_INPUT_ANY_NONZERO) == 0) ||
      (host_validation_flags & SQ_CUDA_INPUT_BAD_ZERO_SCALE) != 0) {
    result = SQ_ERR_SCALE;
    goto done;
  }

  if (collect_timings) {
    SQ_CUDA_TRY(error,
                cudaEventRecord(timing_events[SQ_CUDA_TIMING_START], stream),
                goto done);
  }
  SQ_CUDA_TRY(error,
              (cudaError_t)sq_cuda_launch_k3(bit_width, device_codes, count,
                                             device_payload, block_size,
                                             grid_size, stream),
              goto done);
  if (collect_timings) {
    result = finish_kernel_timing(timing_events[SQ_CUDA_TIMING_START],
                                  timing_events[SQ_CUDA_TIMING_STOP], stream,
                                  &timings->k3_ms);
    if (result != SQ_OK) {
      goto done;
    }
  }
  SQ_CUDA_TRY(error, cudaStreamSynchronize(stream), goto done);
  result =
      timed_copy(payload, device_payload, payload_bytes, cudaMemcpyDeviceToHost,
                 stream, timings != NULL ? &timings->d2h_ms : NULL,
                 collect_timings && timings != NULL);
  if (result != SQ_OK) {
    goto done;
  }
  result = SQ_OK;

done:
  result = error != cudaSuccess ? SQ_ERR_CUDA : result;
  (void)cudaFree(device_values);
  (void)cudaFree(device_scale);
  destroy_k1_workspace(&k1_workspace);
  (void)cudaFree(device_words);
  (void)cudaFree(device_validation_flags);
  (void)cudaFree(device_codes);
  (void)cudaFree(device_payload);
  (void)cudaFree(device_status);
  destroy_events(timing_events, SQ_CUDA_TIMING_COUNT);
  if (stream != NULL) {
    (void)cudaStreamDestroy(stream);
  }
  return result;
}

/* Result words that the pipeline copies back; pinned with the payload when the
   transfer policy is pinned. */
struct sq_cuda_bench_host {
  float scale;
  int status;
  uint32_t validation_flags;
};

static cudaError_t allocate_host(void **pointer, size_t bytes,
                                 sq_cuda_transfer_policy transfer_policy) {
  if (transfer_policy == SQ_CUDA_TRANSFER_PINNED) {
    return cudaHostAlloc(pointer, bytes, cudaHostAllocDefault);
  }
  *pointer = malloc(bytes);
  return *pointer != NULL ? cudaSuccess : cudaErrorMemoryAllocation;
}

static void free_host(void *pointer, sq_cuda_transfer_policy transfer_policy) {
  if (pointer == NULL) {
    return;
  }
  if (transfer_policy == SQ_CUDA_TRANSFER_PINNED) {
    (void)cudaFreeHost(pointer);
  } else {
    free(pointer);
  }
}

enum {
  SQ_BENCH_K1_START,
  SQ_BENCH_K1_STOP,
  SQ_BENCH_K2_START,
  SQ_BENCH_K2_STOP,
  SQ_BENCH_K3_START,
  SQ_BENCH_K3_STOP,
  SQ_BENCH_H2D_START,
  SQ_BENCH_H2D_STOP,
  SQ_BENCH_D2H_START,
  SQ_BENCH_D2H_STOP,
  SQ_BENCH_EVENT_COUNT
};

struct sq_cuda_bench_context {
  sq_cuda_transfer_policy transfer_policy;
  cudaStream_t stream;
  cudaEvent_t events[SQ_BENCH_EVENT_COUNT];
  float *device_values;
  float *device_scale;
  float *device_k1_scale;
  sq_cuda_k1_workspace k1_workspace;
  uint32_t *device_words;
  uint32_t *device_validation_flags;
  uint8_t *device_codes;
  uint8_t *device_payload;
  int *device_status;
  uint8_t *host_payload;
  float *host_values;
  const float *h2d_source;
  sq_cuda_bench_host *host;
};

static void destroy_bench_context(sq_cuda_bench_context *context) {
  (void)cudaFree(context->device_values);
  (void)cudaFree(context->device_scale);
  (void)cudaFree(context->device_k1_scale);
  destroy_k1_workspace(&context->k1_workspace);
  (void)cudaFree(context->device_words);
  (void)cudaFree(context->device_validation_flags);
  (void)cudaFree(context->device_codes);
  (void)cudaFree(context->device_payload);
  (void)cudaFree(context->device_status);
  destroy_events(context->events, SQ_BENCH_EVENT_COUNT);
  if (context->stream != NULL) {
    (void)cudaStreamDestroy(context->stream);
  }
  free_host(context->host_payload, context->transfer_policy);
  free_host(context->host_values, context->transfer_policy);
  free_host(context->host, context->transfer_policy);
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof bits);
  return bits;
}

static sq_status check_bench_result(float host_scale, int prescribed_scale_seen,
                                    int host_status,
                                    uint32_t host_validation_flags) {
  if (host_status != SQ_OK &&
      (!prescribed_scale_seen || host_status != SQ_ERR_SCALE_OVERFLOW)) {
    return (sq_status)host_status;
  }
  if ((host_validation_flags & SQ_CUDA_INPUT_NONFINITE) != 0) {
    return SQ_ERR_NONFINITE;
  }
  if (((host_scale == 0.0f) &&
       (host_validation_flags & SQ_CUDA_INPUT_ANY_NONZERO) != 0) ||
      ((host_scale != 0.0f) &&
       (host_validation_flags & SQ_CUDA_INPUT_ANY_NONZERO) == 0) ||
      (host_validation_flags & SQ_CUDA_INPUT_BAD_ZERO_SCALE) != 0) {
    return SQ_ERR_SCALE;
  }
  return SQ_OK;
}

static cudaError_t enqueue_resident(sq_cuda_bench_context *context,
                                    uint8_t bit_width, size_t count,
                                    int prescribed_scale_seen,
                                    int prescribed_words_seen,
                                    sq_rng_stream stream_state, int block_size,
                                    int grid_size, cudaEvent_t *events) {
  cudaError_t error;

  if (count == 0) {
    return cudaSuccess;
  }
  SQ_CUDA_TRY(error,
              cudaMemsetAsync(context->device_validation_flags, 0,
                              sizeof(uint32_t), context->stream),
              return error);
  if (events != NULL) {
    SQ_CUDA_TRY(error,
                cudaEventRecord(events[SQ_BENCH_K1_START], context->stream),
                return error);
  }
  SQ_CUDA_TRY(error,
              (cudaError_t)launch_k1(
                  context->device_values, count, &context->k1_workspace,
                  prescribed_scale_seen ? context->device_k1_scale
                                        : context->device_scale,
                  context->device_status, grid_size, context->stream),
              return error);
  if (events != NULL) {
    SQ_CUDA_TRY(error,
                cudaEventRecord(events[SQ_BENCH_K1_STOP], context->stream),
                return error);
    SQ_CUDA_TRY(error,
                cudaEventRecord(events[SQ_BENCH_K2_START], context->stream),
                return error);
  }
  SQ_CUDA_TRY(error,
              (cudaError_t)sq_cuda_launch_k2(
                  bit_width, context->device_values, count,
                  context->device_scale, context->device_words,
                  prescribed_words_seen, stream_state, context->device_codes,
                  context->device_validation_flags, block_size, grid_size,
                  context->stream),
              return error);
  if (events != NULL) {
    SQ_CUDA_TRY(error,
                cudaEventRecord(events[SQ_BENCH_K2_STOP], context->stream),
                return error);
    SQ_CUDA_TRY(error,
                cudaEventRecord(events[SQ_BENCH_K3_START], context->stream),
                return error);
  }
  SQ_CUDA_TRY(error,
              (cudaError_t)sq_cuda_launch_k3(bit_width, context->device_codes,
                                             count, context->device_payload,
                                             block_size, grid_size,
                                             context->stream),
              return error);
  if (events != NULL) {
    SQ_CUDA_TRY(error,
                cudaEventRecord(events[SQ_BENCH_K3_STOP], context->stream),
                return error);
  }
  return cudaSuccess;
}

static sq_status
run_bench_pipeline(sq_cuda_bench_context *context, uint8_t bit_width,
                   size_t count, uint64_t seed, uint64_t tensor_id,
                   uint64_t invocation_id, int prescribed_scale_seen,
                   int prescribed_words_seen, int block_size, int grid_size,
                   sq_cuda_bench_boundary boundary, int copy_outputs,
                   int inspect_result, sq_bench_sample *sample) {
  const size_t payload_bytes =
      bit_width == SQ_Q4_BITS ? count / 2 + (count & 1) : count;
  sq_rng_stream stream_state;
  const auto wall_start = std::chrono::steady_clock::now();
  cudaError_t error;

  if (sq_rng_stream_init(&stream_state, seed, tensor_id, invocation_id) !=
      SQ_RNG_OK) {
    return SQ_ERR_ID_OVERFLOW;
  }
  if (boundary == SQ_CUDA_BENCH_HOST_ORIGIN) {
    SQ_CUDA_TRY(
        error,
        cudaEventRecord(context->events[SQ_BENCH_H2D_START], context->stream),
        return SQ_ERR_CUDA);
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(context->device_values, context->h2d_source,
                                count * sizeof(float), cudaMemcpyHostToDevice,
                                context->stream),
                return SQ_ERR_CUDA);
    SQ_CUDA_TRY(
        error,
        cudaEventRecord(context->events[SQ_BENCH_H2D_STOP], context->stream),
        return SQ_ERR_CUDA);
  }

  SQ_CUDA_TRY(error,
              enqueue_resident(context, bit_width, count, prescribed_scale_seen,
                               prescribed_words_seen, stream_state, block_size,
                               grid_size, context->events),
              return SQ_ERR_CUDA);

  if (copy_outputs) {
    SQ_CUDA_TRY(
        error,
        cudaEventRecord(context->events[SQ_BENCH_D2H_START], context->stream),
        return SQ_ERR_CUDA);
    if (payload_bytes != 0) {
      SQ_CUDA_TRY(error,
                  cudaMemcpyAsync(context->host_payload,
                                  context->device_payload, payload_bytes,
                                  cudaMemcpyDeviceToHost, context->stream),
                  return SQ_ERR_CUDA);
    }
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(&context->host->scale, context->device_scale,
                                sizeof(float), cudaMemcpyDeviceToHost,
                                context->stream),
                return SQ_ERR_CUDA);
    SQ_CUDA_TRY(
        error,
        cudaEventRecord(context->events[SQ_BENCH_D2H_STOP], context->stream),
        return SQ_ERR_CUDA);
  }
  if (inspect_result) {
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(&context->host->status, context->device_status,
                                sizeof(int), cudaMemcpyDeviceToHost,
                                context->stream),
                return SQ_ERR_CUDA);
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(&context->host->validation_flags,
                                context->device_validation_flags,
                                sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                context->stream),
                return SQ_ERR_CUDA);
  }

  SQ_CUDA_TRY(error, cudaDeviceSynchronize(), return SQ_ERR_CUDA);
  if (sample != NULL) {
    const auto wall_stop = std::chrono::steady_clock::now();
    float elapsed = 0.0f;
    sample->wall_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start)
            .count();
    SQ_CUDA_TRY(error,
                cudaEventElapsedTime(&elapsed,
                                     context->events[SQ_BENCH_K1_START],
                                     context->events[SQ_BENCH_K1_STOP]),
                return SQ_ERR_CUDA);
    sample->k1_ms = elapsed;
    SQ_CUDA_TRY(error,
                cudaEventElapsedTime(&elapsed,
                                     context->events[SQ_BENCH_K2_START],
                                     context->events[SQ_BENCH_K2_STOP]),
                return SQ_ERR_CUDA);
    sample->k2_ms = elapsed;
    SQ_CUDA_TRY(error,
                cudaEventElapsedTime(&elapsed,
                                     context->events[SQ_BENCH_K3_START],
                                     context->events[SQ_BENCH_K3_STOP]),
                return SQ_ERR_CUDA);
    sample->k3_ms = elapsed;
    sample->h2d_ms = 0.0;
    sample->d2h_ms = 0.0;
    sample->cpu_ms = 0.0;
    if (boundary == SQ_CUDA_BENCH_HOST_ORIGIN) {
      SQ_CUDA_TRY(error,
                  cudaEventElapsedTime(&elapsed,
                                       context->events[SQ_BENCH_H2D_START],
                                       context->events[SQ_BENCH_H2D_STOP]),
                  return SQ_ERR_CUDA);
      sample->h2d_ms = elapsed;
    }
    if (copy_outputs) {
      SQ_CUDA_TRY(error,
                  cudaEventElapsedTime(&elapsed,
                                       context->events[SQ_BENCH_D2H_START],
                                       context->events[SQ_BENCH_D2H_STOP]),
                  return SQ_ERR_CUDA);
      sample->d2h_ms = elapsed;
    }
  }

  if (inspect_result) {
    return check_bench_result(context->host->scale, prescribed_scale_seen,
                              context->host->status,
                              context->host->validation_flags);
  }
  return SQ_OK;
}

/*
 * Captures the resident sequence (1 memset, then K1-K3) once, outside timing,
 * and times each run as one graph launch plus device synchronization. The
 * captured K2 arguments fix the base invocation's RNG stream, so every launch
 * repeats the base workload. After timing, the last launch's outputs must match
 * the uncaptured preflight record byte for byte.
 */
static sq_status run_bench_graph(sq_cuda_bench_context *context,
                                 uint8_t bit_width, size_t count, uint64_t seed,
                                 uint64_t tensor_id, uint64_t invocation_id,
                                 int prescribed_scale_seen,
                                 int prescribed_words_seen, int block_size,
                                 int grid_size, uint64_t warmups, uint64_t reps,
                                 const uint8_t *base_payload, float base_scale,
                                 sq_bench_sample *samples, double *capture_ms) {
  const size_t payload_bytes =
      bit_width == SQ_Q4_BITS ? count / 2 + (count & 1) : count;
  uint64_t index;
  uint32_t host_validation_flags = 0;
  int host_status = SQ_OK;
  sq_rng_stream stream_state;
  cudaGraph_t graph = NULL;
  cudaGraphExec_t graph_exec = NULL;
  sq_status result = SQ_ERR_CUDA;
  cudaError_t error, capture_error = cudaSuccess;

  if (sq_rng_stream_init(&stream_state, seed, tensor_id, invocation_id) !=
      SQ_RNG_OK) {
    return SQ_ERR_ID_OVERFLOW;
  }
  /* Poison the payload left by the preflight so parity proves K3 ran in the
   * graph. */
  if (payload_bytes != 0) {
    for (index = 0; index < payload_bytes; index++) {
      context->host_payload[index] = (uint8_t)~base_payload[index];
    }
    /* A pageable copy can return before its DMA lands; the graph stream does
     * not wait for it. */
    SQ_CUDA_TRY(error,
                cudaMemcpy(context->device_payload, context->host_payload,
                           payload_bytes, cudaMemcpyHostToDevice),
                return SQ_ERR_CUDA);
    SQ_CUDA_TRY(error, cudaDeviceSynchronize(), return SQ_ERR_CUDA);
  }

  const auto capture_start = std::chrono::steady_clock::now();
  SQ_CUDA_TRY(
      error,
      cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeGlobal),
      return SQ_ERR_CUDA);
  if (count != 0) {
    capture_error = enqueue_resident(
        context, bit_width, count, prescribed_scale_seen, prescribed_words_seen,
        stream_state, block_size, grid_size, NULL);
  }
  /* End the capture even after a failed launch so the stream leaves capture
   * mode. */
  SQ_CUDA_TRY(error, cudaStreamEndCapture(context->stream, &graph), goto done);
  if (capture_error != cudaSuccess) {
    goto done;
  }
  SQ_CUDA_TRY(error, cudaGraphInstantiate(&graph_exec, graph, 0), goto done);
  if (capture_ms != NULL) {
    *capture_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - capture_start)
                      .count();
  }

  for (index = 0; index < warmups + reps; index++) {
    sq_bench_sample *sample =
        index < warmups ? NULL : &samples[index - warmups];
    const auto wall_start = std::chrono::steady_clock::now();
    SQ_CUDA_TRY(error, cudaGraphLaunch(graph_exec, context->stream), goto done);
    SQ_CUDA_TRY(error, cudaDeviceSynchronize(), goto done);
    if (sample != NULL) {
      const auto wall_stop = std::chrono::steady_clock::now();
      sample->wall_ms =
          std::chrono::duration<double, std::milli>(wall_stop - wall_start)
              .count();
      sample->k1_ms = 0.0;
      sample->k2_ms = 0.0;
      sample->k3_ms = 0.0;
      sample->h2d_ms = 0.0;
      sample->d2h_ms = 0.0;
      sample->cpu_ms = 0.0;
    }
  }

  if (count != 0) {
    if (payload_bytes != 0) {
      SQ_CUDA_TRY(error,
                  cudaMemcpyAsync(context->host_payload,
                                  context->device_payload, payload_bytes,
                                  cudaMemcpyDeviceToHost, context->stream),
                  goto done);
    }
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(&context->host->scale, context->device_scale,
                                sizeof(float), cudaMemcpyDeviceToHost,
                                context->stream),
                goto done);
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(&host_status, context->device_status,
                                sizeof(int), cudaMemcpyDeviceToHost,
                                context->stream),
                goto done);
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(
                    &host_validation_flags, context->device_validation_flags,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, context->stream),
                goto done);
    SQ_CUDA_TRY(error, cudaDeviceSynchronize(), goto done);
    result = check_bench_result(context->host->scale, prescribed_scale_seen,
                                host_status, host_validation_flags);
    if (result != SQ_OK) {
      goto done;
    }
    result = SQ_ERR_CUDA;
    if (f32_bits(context->host->scale) != f32_bits(base_scale) ||
        (payload_bytes != 0 &&
         memcmp(context->host_payload, base_payload, payload_bytes) != 0)) {
      goto done;
    }
  }
  result = SQ_OK;

done:
  if (graph_exec != NULL) {
    (void)cudaGraphExecDestroy(graph_exec);
  }
  if (graph != NULL) {
    (void)cudaGraphDestroy(graph);
  }
  return result;
}

sq_status sq_cuda_bench(uint8_t bit_width, const float *values, size_t count,
                        uint64_t seed, uint64_t tensor_id,
                        uint64_t base_invocation_id, int prescribed_scale_seen,
                        float prescribed_scale,
                        const uint32_t *prescribed_words, int block_size,
                        int grid_size, sq_cuda_bench_boundary boundary,
                        sq_cuda_transfer_policy transfer_policy,
                        uint64_t warmups, uint64_t reps, uint8_t *base_payload,
                        float *base_scale, sq_bench_sample *samples,
                        double *capture_ms) {
  sq_cuda_bench_context context = {};
  const int transfers = boundary == SQ_CUDA_BENCH_HOST_ORIGIN ||
                        boundary == SQ_CUDA_BENCH_GPU_ORIGIN;
  sq_status result = SQ_ERR_CUDA;
  cudaError_t error;
  uint64_t total_runs, index;
  size_t payload_bytes;

  if (capture_ms != NULL) {
    *capture_ms = 0.0;
  }
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  if ((count != 0 && (values == NULL || base_payload == NULL)) ||
      base_scale == NULL || samples == NULL || reps == 0 ||
      count > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(uint32_t)) {
    return SQ_ERR_ARGUMENT;
  }
  if (!valid_launch_geometry(block_size, grid_size) ||
      (boundary != SQ_CUDA_BENCH_RESIDENT &&
       boundary != SQ_CUDA_BENCH_HOST_ORIGIN &&
       boundary != SQ_CUDA_BENCH_RESIDENT_GRAPH &&
       boundary != SQ_CUDA_BENCH_GPU_ORIGIN) ||
      (transfer_policy != SQ_CUDA_TRANSFER_PAGEABLE &&
       transfer_policy != SQ_CUDA_TRANSFER_PINNED) ||
      (!transfers && transfer_policy != SQ_CUDA_TRANSFER_PAGEABLE)) {
    return SQ_ERR_ARGUMENT;
  }
  if (prescribed_scale_seen &&
      (!std::isfinite(prescribed_scale) || prescribed_scale < 0.0f)) {
    return SQ_ERR_SCALE;
  }
  if (tensor_id > UINT32_MAX || base_invocation_id > UINT32_MAX ||
      warmups > UINT64_MAX - reps) {
    return SQ_ERR_ID_OVERFLOW;
  }
  total_runs = warmups + reps;
  if (total_runs == 0 ||
      total_runs - 1 > (uint64_t)UINT32_MAX - base_invocation_id) {
    return SQ_ERR_ID_OVERFLOW;
  }
  if (prescribed_scale_seen && count == 0 && prescribed_scale != 0.0f) {
    return SQ_ERR_SCALE;
  }

  if (!cuda_device_available()) {
    return SQ_ERR_CUDA;
  }
  SQ_CUDA_TRY(error,
              cudaStreamCreateWithFlags(&context.stream, cudaStreamNonBlocking),
              goto done);
  error = create_events(context.events, SQ_BENCH_EVENT_COUNT);
  if (error != cudaSuccess) {
    goto done;
  }

  payload_bytes = bit_width == SQ_Q4_BITS ? count / 2 + (count & 1) : count;
  context.transfer_policy = transfer_policy;
  context.h2d_source = values;
  if (allocate_host(reinterpret_cast<void **>(&context.host_payload),
                    payload_bytes == 0 ? 1 : payload_bytes,
                    transfer_policy) != cudaSuccess ||
      allocate_host(reinterpret_cast<void **>(&context.host),
                    sizeof *context.host, transfer_policy) != cudaSuccess) {
    result = SQ_ERR_MEMORY;
    goto done;
  }
  context.host->scale = 0.0f;
  context.host->status = SQ_OK;
  context.host->validation_flags = 0;
  /* Pinned host-origin stages the caller's input in page-locked memory once,
   * outside timing. */
  if (boundary == SQ_CUDA_BENCH_HOST_ORIGIN &&
      transfer_policy == SQ_CUDA_TRANSFER_PINNED && count != 0) {
    if (allocate_host(reinterpret_cast<void **>(&context.host_values),
                      count * sizeof(float), transfer_policy) != cudaSuccess) {
      result = SQ_ERR_MEMORY;
      goto done;
    }
    memcpy(context.host_values, values, count * sizeof(float));
    context.h2d_source = context.host_values;
  }
  if (count != 0) {
    SQ_CUDA_TRY(error,
                cudaMalloc(reinterpret_cast<void **>(&context.device_values),
                           count * sizeof(float)),
                goto done);
    SQ_CUDA_TRY(error,
                cudaMalloc(reinterpret_cast<void **>(&context.device_scale),
                           sizeof(float)),
                goto done);
    SQ_CUDA_TRY(error,
                cudaMalloc(reinterpret_cast<void **>(&context.device_status),
                           sizeof(int)),
                goto done);
    SQ_CUDA_TRY(
        error,
        cudaMalloc(reinterpret_cast<void **>(&context.device_validation_flags),
                   sizeof(uint32_t)),
        goto done);
    SQ_CUDA_TRY(error,
                cudaMalloc(reinterpret_cast<void **>(&context.device_codes),
                           count * sizeof(uint8_t)),
                goto done);
    SQ_CUDA_TRY(
        error,
        cudaMalloc(reinterpret_cast<void **>(&context.device_payload),
                   (payload_bytes == 0 ? 1 : payload_bytes) * sizeof(uint8_t)),
        goto done);
    if (prescribed_scale_seen) {
      SQ_CUDA_TRY(
          error,
          cudaMalloc(reinterpret_cast<void **>(&context.device_k1_scale),
                     sizeof(float)),
          goto done);
    }
    if (prescribed_words != NULL) {
      SQ_CUDA_TRY(error,
                  cudaMalloc(reinterpret_cast<void **>(&context.device_words),
                             count * sizeof(uint32_t)),
                  goto done);
    }
    result = set_k1_workspace_geometry(&context.k1_workspace, count);
    if (result != SQ_OK) {
      goto done;
    }
    error = allocate_k1_workspace(&context.k1_workspace);
    if (error != cudaSuccess) {
      goto done;
    }
    if (prescribed_scale_seen) {
      SQ_CUDA_TRY(error,
                  cudaMemcpyAsync(context.device_scale, &prescribed_scale,
                                  sizeof(float), cudaMemcpyHostToDevice,
                                  context.stream),
                  goto done);
    }
    if (prescribed_words != NULL) {
      SQ_CUDA_TRY(error,
                  cudaMemcpyAsync(context.device_words, prescribed_words,
                                  count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, context.stream),
                  goto done);
    }
    if (boundary == SQ_CUDA_BENCH_RESIDENT ||
        boundary == SQ_CUDA_BENCH_RESIDENT_GRAPH ||
        boundary == SQ_CUDA_BENCH_GPU_ORIGIN) {
      SQ_CUDA_TRY(error,
                  cudaMemcpyAsync(context.device_values, values,
                                  count * sizeof(float), cudaMemcpyHostToDevice,
                                  context.stream),
                  goto done);
    }
    SQ_CUDA_TRY(error, cudaStreamSynchronize(context.stream), goto done);
  }

  if (count == 0) {
    *base_scale = 0.0f;
    if (boundary == SQ_CUDA_BENCH_RESIDENT_GRAPH) {
      result = run_bench_graph(&context, bit_width, count, seed, tensor_id,
                               base_invocation_id, prescribed_scale_seen,
                               prescribed_words != NULL, block_size, grid_size,
                               warmups, reps, base_payload, *base_scale,
                               samples, capture_ms);
      goto done;
    }
    for (index = 0; index < warmups + reps; index++) {
      sq_bench_sample *sample =
          index < warmups ? NULL : &samples[index - warmups];
      const auto start = std::chrono::steady_clock::now();
      SQ_CUDA_TRY(error, cudaDeviceSynchronize(), goto done);
      if (sample != NULL) {
        const auto stop = std::chrono::steady_clock::now();
        sample->wall_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        sample->k1_ms = 0.0;
        sample->k2_ms = 0.0;
        sample->k3_ms = 0.0;
        sample->h2d_ms = 0.0;
        sample->d2h_ms = 0.0;
        sample->cpu_ms = 0.0;
      }
    }
    result = SQ_OK;
    goto done;
  }

  result = run_bench_pipeline(
      &context, bit_width, count, seed, tensor_id, base_invocation_id,
      prescribed_scale_seen, prescribed_words != NULL, block_size, grid_size,
      boundary == SQ_CUDA_BENCH_HOST_ORIGIN ? SQ_CUDA_BENCH_HOST_ORIGIN
                                            : SQ_CUDA_BENCH_RESIDENT,
      1, 1, NULL);
  if (result != SQ_OK) {
    goto done;
  }
  if (payload_bytes != 0) {
    memcpy(base_payload, context.host_payload, payload_bytes);
  }
  *base_scale = context.host->scale;

  if (boundary == SQ_CUDA_BENCH_RESIDENT_GRAPH) {
    result = run_bench_graph(
        &context, bit_width, count, seed, tensor_id, base_invocation_id,
        prescribed_scale_seen, prescribed_words != NULL, block_size, grid_size,
        warmups, reps, base_payload, *base_scale, samples, capture_ms);
    goto done;
  }

  for (index = 0; index < total_runs; index++) {
    sq_bench_sample *sample =
        index < warmups ? NULL : &samples[index - warmups];
    const uint64_t run_offset =
        index < warmups ? reps + index : index - warmups;
    const uint64_t invocation_id = base_invocation_id + run_offset;
    result = run_bench_pipeline(&context, bit_width, count, seed, tensor_id,
                                invocation_id, prescribed_scale_seen,
                                prescribed_words != NULL, block_size, grid_size,
                                boundary, transfers, 0, sample);
    if (result != SQ_OK) {
      goto done;
    }
  }
  result = SQ_OK;

done:
  destroy_bench_context(&context);
  return result;
}

/*
 * GPU-origin staging for the CPU backend: the input lives on the device, and
 * each download times one full D2H into a landing buffer of the chosen policy.
 */
struct sq_cuda_staging {
  sq_cuda_transfer_policy transfer_policy;
  cudaStream_t stream;
  cudaEvent_t events[SQ_CUDA_TIMING_COUNT];
  float *device_values;
  float *landing;
  size_t count;
};

void sq_cuda_staging_destroy(sq_cuda_staging *staging) {
  if (staging == NULL) {
    return;
  }
  if (staging->device_values != NULL) {
    (void)cudaFree(staging->device_values);
  }
  free_host(staging->landing, staging->transfer_policy);
  destroy_events(staging->events, SQ_CUDA_TIMING_COUNT);
  if (staging->stream != NULL) {
    (void)cudaStreamDestroy(staging->stream);
  }
  free(staging);
}

sq_status sq_cuda_staging_create(const float *values, size_t count,
                                 sq_cuda_transfer_policy transfer_policy,
                                 sq_cuda_staging **staging) {
  sq_cuda_staging *created;
  cudaError_t error;

  if (staging == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  *staging = NULL;
  if ((count != 0 && values == NULL) || count > SIZE_MAX / sizeof(float) ||
      (transfer_policy != SQ_CUDA_TRANSFER_PAGEABLE &&
       transfer_policy != SQ_CUDA_TRANSFER_PINNED)) {
    return SQ_ERR_ARGUMENT;
  }
  if (!cuda_device_available()) {
    return SQ_ERR_CUDA;
  }
  created = (sq_cuda_staging *)calloc(1, sizeof *created);
  if (created == NULL) {
    return SQ_ERR_MEMORY;
  }
  created->transfer_policy = transfer_policy;
  created->count = count;
  if (allocate_host(reinterpret_cast<void **>(&created->landing),
                    count == 0 ? 1 : count * sizeof(float),
                    transfer_policy) != cudaSuccess) {
    sq_cuda_staging_destroy(created);
    return SQ_ERR_MEMORY;
  }
  SQ_CUDA_TRY(
      error, cudaStreamCreateWithFlags(&created->stream, cudaStreamNonBlocking),
      {
        sq_cuda_staging_destroy(created);
        return SQ_ERR_CUDA;
      });
  error = create_events(created->events, SQ_CUDA_TIMING_COUNT);
  if (error != cudaSuccess) {
    sq_cuda_staging_destroy(created);
    return SQ_ERR_CUDA;
  }
  SQ_CUDA_TRY(error,
              cudaMalloc(reinterpret_cast<void **>(&created->device_values),
                         count == 0 ? 1 : count * sizeof(float)),
              {
                sq_cuda_staging_destroy(created);
                return SQ_ERR_CUDA;
              });
  if (count != 0) {
    SQ_CUDA_TRY(error,
                cudaMemcpy(created->device_values, values,
                           count * sizeof(float), cudaMemcpyHostToDevice),
                {
                  sq_cuda_staging_destroy(created);
                  return SQ_ERR_CUDA;
                });
  }
  SQ_CUDA_TRY(error, cudaDeviceSynchronize(), {
    sq_cuda_staging_destroy(created);
    return SQ_ERR_CUDA;
  });
  *staging = created;
  return SQ_OK;
}

sq_status sq_cuda_staging_download(sq_cuda_staging *staging,
                                   const float **landing, double *d2h_ms) {
  float elapsed = 0.0f;

  if (staging == NULL || landing == NULL || d2h_ms == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  cudaError_t error;

  SQ_CUDA_TRY(
      error,
      cudaEventRecord(staging->events[SQ_CUDA_TIMING_START], staging->stream),
      return SQ_ERR_CUDA);
  if (staging->count != 0) {
    SQ_CUDA_TRY(error,
                cudaMemcpyAsync(staging->landing, staging->device_values,
                                staging->count * sizeof(float),
                                cudaMemcpyDeviceToHost, staging->stream),
                return SQ_ERR_CUDA);
  }
  SQ_CUDA_TRY(
      error,
      cudaEventRecord(staging->events[SQ_CUDA_TIMING_STOP], staging->stream),
      return SQ_ERR_CUDA);
  SQ_CUDA_TRY(error, cudaStreamSynchronize(staging->stream),
              return SQ_ERR_CUDA);
  SQ_CUDA_TRY(error,
              cudaEventElapsedTime(&elapsed,
                                   staging->events[SQ_CUDA_TIMING_START],
                                   staging->events[SQ_CUDA_TIMING_STOP]),
              return SQ_ERR_CUDA);
  *landing = staging->landing;
  *d2h_ms = elapsed;
  return SQ_OK;
}
