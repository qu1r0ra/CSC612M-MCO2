#include "quantizer_cuda.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#define MCO2_CUDA_REDUCTION_THREADS 256
#define MCO2_CUDA_MAX_AUTO_GRID MCO2_CUDA_MAX_GRID_SIZE

enum {
    MCO2_CUDA_INPUT_NONFINITE = 1U,
    MCO2_CUDA_INPUT_ANY_NONZERO = 2U,
    MCO2_CUDA_INPUT_BAD_ZERO_SCALE = 4U
};

__global__ static void max_blocks_kernel(const float *values, uint64_t count,
                                         uint64_t block_count,
                                         float *max_partials,
                                         uint32_t *invalid_partials)
{
    __shared__ float block_max[MCO2_CUDA_REDUCTION_THREADS];
    __shared__ uint32_t block_invalid[MCO2_CUDA_REDUCTION_THREADS];
    const unsigned int lane = threadIdx.x;
    uint64_t logical_block;

    for (logical_block = blockIdx.x; logical_block < block_count;
         logical_block += gridDim.x) {
        const uint64_t index = logical_block * MCO2_CUDA_REDUCTION_THREADS + lane;
        float value_max = 0.0f;
        uint32_t invalid = 0;

        if (index < count) {
            const float value = values[index];
            if (isfinite(value))
                value_max = fabsf(value);
            else
                invalid = 1;
        }
        block_max[lane] = value_max;
        block_invalid[lane] = invalid;
        __syncthreads();

        for (unsigned int stride = MCO2_CUDA_REDUCTION_THREADS / 2;
             stride != 0; stride /= 2) {
            if (lane < stride) {
                const float other_max = block_max[lane + stride];
                if (other_max > block_max[lane])
                    block_max[lane] = other_max;
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
                                         int *status)
{
    __shared__ float maxima[MCO2_CUDA_REDUCTION_THREADS];
    __shared__ uint32_t invalid[MCO2_CUDA_REDUCTION_THREADS];
    const unsigned int lane = threadIdx.x;
    float local_max = 0.0f;
    uint32_t local_invalid = 0;

    for (uint64_t i = lane; i < block_count; i += MCO2_CUDA_REDUCTION_THREADS) {
        if (max_partials[i] > local_max)
            local_max = max_partials[i];
        local_invalid |= invalid_partials[i];
    }
    maxima[lane] = local_max;
    invalid[lane] = local_invalid;
    __syncthreads();

    for (unsigned int stride = MCO2_CUDA_REDUCTION_THREADS / 2; stride != 0;
         stride /= 2) {
        if (lane < stride) {
            if (maxima[lane + stride] > maxima[lane])
                maxima[lane] = maxima[lane + stride];
            invalid[lane] |= invalid[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        *scale = maxima[0];
        *status = invalid[0] ? MCO2_Q8_ERR_NONFINITE : MCO2_Q8_OK;
    }
}

__global__ static void sum_blocks_kernel(const float *values, uint64_t count,
                                         uint64_t block_count,
                                         uint64_t padded_count,
                                         const float *max_abs, const int *status,
                                         float *partials)
{
    __shared__ float terms[2][MCO2_CUDA_REDUCTION_THREADS];
    const unsigned int lane = threadIdx.x;
    uint64_t logical_block;

    for (logical_block = blockIdx.x; logical_block < padded_count;
         logical_block += gridDim.x) {
        if (logical_block >= block_count || *status != MCO2_Q8_OK || *max_abs == 0.0f) {
            if (lane == 0)
                partials[logical_block] = 0.0f;
            __syncthreads();
            continue;
        }

        const uint64_t index = logical_block * MCO2_CUDA_REDUCTION_THREADS + lane;
        float term = 0.0f;
        if (index < count) {
            const float ratio = fabsf(values[index]) / *max_abs;
            term = ratio * ratio;
        }
        float *input = terms[0];
        float *output = terms[1];
        input[lane] = term;
        __syncthreads();

        for (unsigned int stride = MCO2_CUDA_REDUCTION_THREADS / 2;
             stride != 0; stride /= 2) {
            if (lane < stride)
                output[lane] = input[2 * lane] + input[2 * lane + 1];
            __syncthreads();
            float *temporary = input;
            input = output;
            output = temporary;
        }
        if (lane == 0)
            partials[logical_block] = input[0];
        __syncthreads();
    }
}

__global__ static void reduce_pairs_kernel(const float *input, float *output,
                                           uint64_t output_count)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < output_count; i += stride)
        output[i] = input[2 * i] + input[2 * i + 1];
}

__global__ static void finish_scale_kernel(const float *sum, float *scale,
                                           int *status)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || *status != MCO2_Q8_OK)
        return;
    if (*scale == 0.0f)
        return;

    const float root = sqrtf(*sum);
    const float result = *scale * root;
    if (!isfinite(result)) {
        *status = MCO2_Q8_ERR_SCALE_OVERFLOW;
        *scale = 0.0f;
    } else {
        *scale = result;
    }
}

__global__ static void round_kernel(uint8_t bit_width,
                                    const float *values, uint64_t count,
                                    const float *scale,
                                    const uint32_t *words,
                                    int prescribed_words,
                                    mco2_rng_stream stream_state,
                                    uint8_t *codes,
                                    uint32_t *validation_flags)
{
    const int s = (bit_width == MCO2_Q4_BITS) ? MCO2_Q4_SIGNED_LIMIT : MCO2_Q8_SIGNED_LIMIT;
    const philox4x32_key_t key = mco2_philox_key(&stream_state);
    const uint64_t group_count = count / 4 + (count % 4 != 0);
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const float scale_value = *scale;

    for (uint64_t group = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         group < group_count; group += stride) {
        philox4x32_ctr_t random_words;
        if (!prescribed_words) {
            random_words = philox4x32_R(MCO2_PHILOX_ROUNDS,
                                        mco2_philox_ctr(&stream_state, group), key);
        }
        for (unsigned int lane = 0; lane < 4; lane++) {
            const uint64_t index = group * 4 + lane;
            if (index >= count)
                break;

            const float value = values[index];
            if (!isfinite(value)) {
                atomicOr(validation_flags, MCO2_CUDA_INPUT_NONFINITE);
                continue;
            }
            if (value != 0.0f)
                atomicOr(validation_flags, MCO2_CUDA_INPUT_ANY_NONZERO);
            if (scale_value == 0.0f) {
                if (value != 0.0f)
                    atomicOr(validation_flags, MCO2_CUDA_INPUT_BAD_ZERO_SCALE);
                codes[index] = (uint8_t)s;
                continue;
            }

            const float absolute_value = fabsf(value);
            float scaled = (absolute_value / scale_value) * (float)s;
            if (scaled > (float)s)
                scaled = (float)s;
            const float lower_float = floorf(scaled);
            const float probability = scaled - lower_float;
            const uint32_t word = prescribed_words ? words[index] : random_words.v[lane];
            const int magnitude = (int)lower_float + mco2_bernoulli(word, probability);
            const int signed_code = signbit(value) && magnitude != 0 ? -magnitude : magnitude;
            codes[index] = (uint8_t)(signed_code + s);
        }
    }
}

__global__ static void pack_q8_kernel(const uint8_t *codes, uint64_t count,
                                      uint8_t *payload)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride)
        payload[index] = codes[index];
}

__global__ static void pack_q4_kernel(const uint8_t *codes, uint64_t count,
                                      uint64_t output_bytes, uint8_t *payload)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t byte_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         byte_index < output_bytes; byte_index += stride) {
        const uint64_t idx0 = byte_index * 2;
        const uint64_t idx1 = idx0 + 1;
        const uint8_t low_nibble = (uint8_t)(codes[idx0] & 0x0F);
        const uint8_t high_nibble = (idx1 < count) ? (uint8_t)(codes[idx1] & 0x0F) : (uint8_t)0;
        payload[byte_index] = (uint8_t)(low_nibble | (high_nibble << 4));
    }
}

static int automatic_grid(uint64_t work, int block_size)
{
    const uint64_t blocks = work / (uint64_t)block_size +
                            (work % (uint64_t)block_size != 0);
    return (int)(blocks < MCO2_CUDA_MAX_AUTO_GRID ? blocks : MCO2_CUDA_MAX_AUTO_GRID);
}

static int launch_grid(uint64_t work, int block_size, int grid_size)
{
    if (block_size <= 0)
        return 0;
    if (grid_size > 0)
        return grid_size;
    return automatic_grid(work, block_size);
}

extern "C" int mco2_cuda_launch_k1(const float *device_values,
                                   uint64_t count,
                                   float *device_max_partials,
                                   uint32_t *device_invalid_partials,
                                   float *device_sums_a,
                                   float *device_sums_b,
                                   uint64_t block_count,
                                   uint64_t padded_count,
                                   float *device_scale,
                                   int *device_status,
                                   int grid_size,
                                   void *stream_handle)
{
    if (grid_size < 0 || grid_size > MCO2_CUDA_MAX_GRID_SIZE)
        return (int)cudaErrorInvalidValue;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const uint64_t expected_block_count =
        count / MCO2_CUDA_REDUCTION_THREADS +
        (count % MCO2_CUDA_REDUCTION_THREADS != 0);
    uint64_t expected_padded_count = 1;
    while (expected_padded_count < expected_block_count)
        expected_padded_count <<= 1;
    const unsigned int auto_max_grid = (unsigned int)(block_count < MCO2_CUDA_MAX_AUTO_GRID
                                                          ? block_count
                                                          : MCO2_CUDA_MAX_AUTO_GRID);
    const unsigned int auto_sum_grid = (unsigned int)(padded_count < MCO2_CUDA_MAX_AUTO_GRID
                                                          ? padded_count
                                                          : MCO2_CUDA_MAX_AUTO_GRID);
    const unsigned int max_grid = (grid_size > 0) ? (unsigned int)grid_size : auto_max_grid;
    const unsigned int sum_grid = (grid_size > 0) ? (unsigned int)grid_size : auto_sum_grid;

    if (count == 0 || block_count != expected_block_count ||
        padded_count != expected_padded_count || max_grid == 0 || sum_grid == 0 ||
        device_values == NULL || device_max_partials == NULL ||
        device_invalid_partials == NULL || device_sums_a == NULL ||
        device_sums_b == NULL || device_scale == NULL || device_status == NULL)
        return (int)cudaErrorInvalidValue;

    max_blocks_kernel<<<max_grid, MCO2_CUDA_REDUCTION_THREADS, 0, stream>>>(
        device_values, count, block_count, device_max_partials,
        device_invalid_partials);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return (int)error;

    reduce_max_kernel<<<1, MCO2_CUDA_REDUCTION_THREADS, 0, stream>>>(
        device_max_partials, device_invalid_partials, block_count, device_scale,
        device_status);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return (int)error;

    sum_blocks_kernel<<<sum_grid, MCO2_CUDA_REDUCTION_THREADS, 0, stream>>>(
        device_values, count, block_count, padded_count, device_scale,
        device_status, device_sums_a);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return (int)error;

    uint64_t active = padded_count;
    float *input = device_sums_a;
    float *output = device_sums_b;
    while (active > 1) {
        const uint64_t output_count = active / 2;
        const unsigned int auto_pair_grid = (unsigned int)automatic_grid(output_count, 256);
        const unsigned int pair_grid = (grid_size > 0) ? (unsigned int)grid_size : auto_pair_grid;
        reduce_pairs_kernel<<<pair_grid, 256, 0, stream>>>(input, output, output_count);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return (int)error;
        active = output_count;
        float *temporary = input;
        input = output;
        output = temporary;
    }

    finish_scale_kernel<<<1, 1, 0, stream>>>(input, device_scale, device_status);
    return (int)cudaGetLastError();
}

extern "C" int mco2_cuda_launch_q8_k1(const float *device_values,
                                      uint64_t count,
                                      float *device_max_partials,
                                      uint32_t *device_invalid_partials,
                                      float *device_sums_a,
                                      float *device_sums_b,
                                      uint64_t block_count,
                                      uint64_t padded_count,
                                      float *device_scale,
                                      int *device_status,
                                      void *stream_handle)
{
    return mco2_cuda_launch_k1(device_values, count, device_max_partials,
                               device_invalid_partials, device_sums_a,
                               device_sums_b, block_count, padded_count,
                               device_scale, device_status, 0, stream_handle);
}

extern "C" int mco2_cuda_launch_k2(uint8_t bit_width,
                                   const float *device_values,
                                   uint64_t count,
                                   const float *device_scale,
                                   const uint32_t *device_words,
                                   int prescribed_words,
                                   mco2_rng_stream stream_state,
                                   uint8_t *device_codes,
                                   uint32_t *device_validation_flags,
                                   int block_size,
                                   int grid_size,
                                   void *stream_handle)
{
    if (bit_width != MCO2_Q4_BITS && bit_width != MCO2_Q8_BITS)
        return (int)cudaErrorInvalidValue;
    if (count == 0)
        return 0;
    if (grid_size < 0 || grid_size > MCO2_CUDA_MAX_GRID_SIZE ||
        block_size <= 0 || block_size > MCO2_CUDA_MAX_BLOCK_SIZE)
        return (int)cudaErrorInvalidValue;
    const uint64_t group_count = count / 4 + (count % 4 != 0);
    const int grid = launch_grid(group_count, block_size, grid_size);
    if (device_values == NULL || device_scale == NULL || device_codes == NULL ||
        device_validation_flags == NULL || grid <= 0 ||
        (prescribed_words && device_words == NULL))
        return (int)cudaErrorInvalidValue;

    round_kernel<<<grid, block_size, 0,
                   reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        bit_width, device_values, count, device_scale, device_words, prescribed_words,
        stream_state, device_codes, device_validation_flags);
    return (int)cudaGetLastError();
}

extern "C" int mco2_cuda_launch_q8_k2(const float *device_values,
                                      uint64_t count,
                                      const float *device_scale,
                                      const uint32_t *device_words,
                                      int prescribed_words,
                                      mco2_rng_stream stream_state,
                                      uint8_t *device_codes,
                                      uint32_t *device_validation_flags,
                                      int block_size,
                                      int grid_size,
                                      void *stream_handle)
{
    return mco2_cuda_launch_k2(MCO2_Q8_BITS, device_values, count, device_scale,
                               device_words, prescribed_words, stream_state,
                               device_codes, device_validation_flags, block_size,
                               grid_size, stream_handle);
}

extern "C" int mco2_cuda_launch_q8_k3(const uint8_t *device_codes,
                                      uint64_t count,
                                      uint8_t *device_payload,
                                      int block_size,
                                      int grid_size,
                                      void *stream_handle)
{
    if (count == 0)
        return 0;
    if (grid_size < 0 || grid_size > MCO2_CUDA_MAX_GRID_SIZE ||
        block_size <= 0 || block_size > MCO2_CUDA_MAX_BLOCK_SIZE)
        return (int)cudaErrorInvalidValue;
    const int grid = launch_grid(count, block_size, grid_size);
    if (device_codes == NULL || device_payload == NULL || grid <= 0)
        return (int)cudaErrorInvalidValue;

    pack_q8_kernel<<<grid, block_size, 0,
                     reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        device_codes, count, device_payload);
    return (int)cudaGetLastError();
}

extern "C" int mco2_cuda_launch_q4_k3(const uint8_t *device_codes,
                                      uint64_t count,
                                      uint8_t *device_payload,
                                      int block_size,
                                      int grid_size,
                                      void *stream_handle)
{
    if (count == 0)
        return 0;
    if (grid_size < 0 || grid_size > MCO2_CUDA_MAX_GRID_SIZE ||
        block_size <= 0 || block_size > MCO2_CUDA_MAX_BLOCK_SIZE)
        return (int)cudaErrorInvalidValue;
    const uint64_t output_bytes = count / 2 + (count & 1);
    const int grid = launch_grid(output_bytes, block_size, grid_size);
    if (device_codes == NULL || device_payload == NULL || grid <= 0)
        return (int)cudaErrorInvalidValue;

    pack_q4_kernel<<<grid, block_size, 0,
                     reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        device_codes, count, output_bytes, device_payload);
    return (int)cudaGetLastError();
}

extern "C" int mco2_cuda_launch_k3(uint8_t bit_width,
                                   const uint8_t *device_codes,
                                   uint64_t count,
                                   uint8_t *device_payload,
                                   int block_size,
                                   int grid_size,
                                   void *stream_handle)
{
    if (bit_width == MCO2_Q8_BITS)
        return mco2_cuda_launch_q8_k3(device_codes, count, device_payload,
                                      block_size, grid_size, stream_handle);
    if (bit_width == MCO2_Q4_BITS)
        return mco2_cuda_launch_q4_k3(device_codes, count, device_payload,
                                      block_size, grid_size, stream_handle);
    return (int)cudaErrorInvalidValue;
}

static mco2_q8_status timed_copy(void *destination, const void *source, size_t size,
                                 cudaMemcpyKind kind, cudaStream_t stream,
                                 float *elapsed_ms, int collect_timing)
{
    const auto start = std::chrono::steady_clock::now();
    cudaError_t error = cudaMemcpyAsync(destination, source, size, kind, stream);
    if (error == cudaSuccess)
        error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    if (collect_timing) {
        const auto end = std::chrono::steady_clock::now();
        *elapsed_ms += std::chrono::duration<float, std::milli>(end - start).count();
    }
    return MCO2_Q8_OK;
}

static mco2_q8_status finish_kernel_timing(cudaEvent_t start, cudaEvent_t stop,
                                           cudaStream_t stream, float *elapsed_ms)
{
    cudaError_t error = cudaEventRecord(stop, stream);
    if (error == cudaSuccess)
        error = cudaEventSynchronize(stop);
    if (error == cudaSuccess)
        error = cudaEventElapsedTime(elapsed_ms, start, stop);
    return error == cudaSuccess ? MCO2_Q8_OK : MCO2_Q8_ERR_CUDA;
}

mco2_q8_status mco2_cuda_compress(uint8_t bit_width, const float *values,
                                  size_t count, uint64_t seed,
                                  uint64_t tensor_id, uint64_t invocation_id,
                                  int prescribed_scale_seen,
                                  float prescribed_scale,
                                  const uint32_t *prescribed_words,
                                  int block_size, int grid_size,
                                  uint8_t *payload, float *scale,
                                  int collect_timings,
                                  mco2_cuda_timings *timings)
{
    cudaStream_t stream = NULL;
    cudaEvent_t event_start = NULL, event_stop = NULL;
    float *device_values = NULL, *device_scale = NULL;
    float *device_max_partials = NULL, *device_sums_a = NULL, *device_sums_b = NULL;
    uint32_t *device_invalid_partials = NULL, *device_words = NULL;
    uint32_t *device_validation_flags = NULL;
    uint8_t *device_codes = NULL, *device_payload = NULL;
    int *device_status = NULL;
    uint64_t block_count = 0, padded_count = 1;
    uint32_t host_validation_flags = 0;
    int host_status = MCO2_Q8_OK;
    int device_count = 0;
    mco2_rng_stream stream_state;
    mco2_q8_status result = MCO2_Q8_ERR_CUDA;
    mco2_cuda_timings zero_timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const size_t payload_bytes = (bit_width == MCO2_Q4_BITS) ? (count / 2 + (count & 1)) : count;

    if (timings != NULL)
        *timings = zero_timings;
    if (bit_width != MCO2_Q4_BITS && bit_width != MCO2_Q8_BITS)
        return MCO2_Q8_ERR_BIT_WIDTH;
    if (scale == NULL || (count != 0 && (values == NULL || payload == NULL)) ||
        count > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(uint32_t) ||
        (collect_timings && timings == NULL))
        return MCO2_Q8_ERR_ARGUMENT;
    if (block_size <= 0 || block_size > MCO2_CUDA_MAX_BLOCK_SIZE ||
        grid_size < 0 || grid_size > MCO2_CUDA_MAX_GRID_SIZE)
        return MCO2_Q8_ERR_ARGUMENT;
    if (prescribed_scale_seen &&
        (!isfinite(prescribed_scale) || prescribed_scale < 0.0f))
        return MCO2_Q8_ERR_SCALE;
    if (mco2_rng_stream_init(&stream_state, seed, tensor_id, invocation_id) != MCO2_OK)
        return MCO2_Q8_ERR_ID_OVERFLOW;

    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        (void)cudaGetLastError();
        return MCO2_Q8_ERR_CUDA;
    }
    if (count == 0 && prescribed_scale_seen && prescribed_scale != 0.0f)
        return MCO2_Q8_ERR_SCALE;

    if (count == 0) {
        *scale = prescribed_scale_seen ? prescribed_scale : 0.0f;
        return MCO2_Q8_OK;
    }

    block_count = count / MCO2_CUDA_REDUCTION_THREADS +
                  (count % MCO2_CUDA_REDUCTION_THREADS != 0);
    while (!prescribed_scale_seen && padded_count < block_count) {
        if (padded_count > std::numeric_limits<uint64_t>::max() / 2)
            return MCO2_Q8_ERR_COUNT;
        padded_count *= 2;
    }

    error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (error != cudaSuccess)
        goto done;
    if (collect_timings) {
        error = cudaEventCreate(&event_start);
        if (error != cudaSuccess)
            goto done;
        error = cudaEventCreate(&event_stop);
        if (error != cudaSuccess)
            goto done;
    }

#define ALLOCATE_DEVICE(pointer, bytes)                                                        \
    do {                                                                                       \
        error = cudaMalloc(reinterpret_cast<void **>(&(pointer)), (bytes));                     \
        if (error != cudaSuccess)                                                              \
            goto done;                                                                         \
    } while (0)

    ALLOCATE_DEVICE(device_values, count * sizeof(float));
    ALLOCATE_DEVICE(device_scale, sizeof(float));
    ALLOCATE_DEVICE(device_status, sizeof(int));
    ALLOCATE_DEVICE(device_validation_flags, sizeof(uint32_t));
    ALLOCATE_DEVICE(device_codes, count * sizeof(uint8_t));
    ALLOCATE_DEVICE(device_payload, (payload_bytes > 0 ? payload_bytes : 1) * sizeof(uint8_t));
    if (prescribed_words != NULL)
        ALLOCATE_DEVICE(device_words, count * sizeof(uint32_t));
    if (!prescribed_scale_seen) {
        ALLOCATE_DEVICE(device_max_partials, block_count * sizeof(float));
        ALLOCATE_DEVICE(device_invalid_partials, block_count * sizeof(uint32_t));
        ALLOCATE_DEVICE(device_sums_a, padded_count * sizeof(float));
        ALLOCATE_DEVICE(device_sums_b, padded_count * sizeof(float));
    }

    error = cudaMemsetAsync(device_status, 0, sizeof(int), stream);
    if (error != cudaSuccess)
        goto done;
    error = cudaMemsetAsync(device_validation_flags, 0, sizeof(uint32_t), stream);
    if (error != cudaSuccess)
        goto done;

    result = timed_copy(device_values, values, count * sizeof(float),
                        cudaMemcpyHostToDevice, stream,
                        timings != NULL ? &timings->h2d_ms : NULL,
                        collect_timings && timings != NULL);
    if (result != MCO2_Q8_OK)
        goto done;
    if (prescribed_words != NULL) {
        result = timed_copy(device_words, prescribed_words, count * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream,
                            timings != NULL ? &timings->h2d_ms : NULL,
                            collect_timings && timings != NULL);
        if (result != MCO2_Q8_OK)
            goto done;
    }

    if (prescribed_scale_seen) {
        *scale = prescribed_scale;
        result = timed_copy(device_scale, scale, sizeof(float),
                            cudaMemcpyHostToDevice, stream,
                            timings != NULL ? &timings->h2d_ms : NULL,
                            collect_timings && timings != NULL);
        if (result != MCO2_Q8_OK)
            goto done;
    } else {
        if (collect_timings) {
            error = cudaEventRecord(event_start, stream);
            if (error != cudaSuccess)
                goto done;
        }
        error = (cudaError_t)mco2_cuda_launch_k1(
            device_values, count, device_max_partials, device_invalid_partials,
            device_sums_a, device_sums_b, block_count, padded_count,
            device_scale, device_status, grid_size, stream);
        if (error != cudaSuccess)
            goto done;
        if (collect_timings) {
            result = finish_kernel_timing(event_start, event_stop, stream,
                                          &timings->k1_ms);
            if (result != MCO2_Q8_OK)
                goto done;
        }
        error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
            goto done;
        result = timed_copy(&host_status, device_status, sizeof(int),
                            cudaMemcpyDeviceToHost, stream,
                            timings != NULL ? &timings->d2h_ms : NULL,
                            collect_timings && timings != NULL);
        if (result != MCO2_Q8_OK)
            goto done;
        result = timed_copy(scale, device_scale, sizeof(float),
                            cudaMemcpyDeviceToHost, stream,
                            timings != NULL ? &timings->d2h_ms : NULL,
                            collect_timings && timings != NULL);
        if (result != MCO2_Q8_OK)
            goto done;
        if (host_status != MCO2_Q8_OK) {
            result = (mco2_q8_status)host_status;
            goto done;
        }
    }

    if (collect_timings) {
        error = cudaEventRecord(event_start, stream);
        if (error != cudaSuccess)
            goto done;
    }
    error = (cudaError_t)mco2_cuda_launch_k2(
        bit_width, device_values, count, device_scale, device_words,
        prescribed_words != NULL, stream_state, device_codes,
        device_validation_flags, block_size, grid_size, stream);
    if (error != cudaSuccess)
        goto done;
    if (collect_timings) {
        result = finish_kernel_timing(event_start, event_stop, stream,
                                      &timings->k2_ms);
        if (result != MCO2_Q8_OK)
            goto done;
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        goto done;
    result = timed_copy(&host_validation_flags, device_validation_flags,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream,
                        timings != NULL ? &timings->d2h_ms : NULL,
                        collect_timings && timings != NULL);
    if (result != MCO2_Q8_OK)
        goto done;
    if ((host_validation_flags & MCO2_CUDA_INPUT_NONFINITE) != 0) {
        result = MCO2_Q8_ERR_NONFINITE;
        goto done;
    }
    if (((*scale == 0.0f) &&
         (host_validation_flags & MCO2_CUDA_INPUT_ANY_NONZERO) != 0) ||
        ((*scale != 0.0f) &&
         (host_validation_flags & MCO2_CUDA_INPUT_ANY_NONZERO) == 0) ||
        (host_validation_flags & MCO2_CUDA_INPUT_BAD_ZERO_SCALE) != 0) {
        result = MCO2_Q8_ERR_SCALE;
        goto done;
    }

    if (collect_timings) {
        error = cudaEventRecord(event_start, stream);
        if (error != cudaSuccess)
            goto done;
    }
    error = (cudaError_t)mco2_cuda_launch_k3(bit_width, device_codes, count,
                                             device_payload, block_size,
                                             grid_size, stream);
    if (error != cudaSuccess)
        goto done;
    if (collect_timings) {
        result = finish_kernel_timing(event_start, event_stop, stream,
                                      &timings->k3_ms);
        if (result != MCO2_Q8_OK)
            goto done;
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        goto done;
    result = timed_copy(payload, device_payload, payload_bytes,
                        cudaMemcpyDeviceToHost, stream,
                        timings != NULL ? &timings->d2h_ms : NULL,
                        collect_timings && timings != NULL);
    if (result != MCO2_Q8_OK)
        goto done;
    result = MCO2_Q8_OK;

done:
    if (device_values != NULL)
        (void)cudaFree(device_values);
    if (device_scale != NULL)
        (void)cudaFree(device_scale);
    if (device_max_partials != NULL)
        (void)cudaFree(device_max_partials);
    if (device_invalid_partials != NULL)
        (void)cudaFree(device_invalid_partials);
    if (device_sums_a != NULL)
        (void)cudaFree(device_sums_a);
    if (device_sums_b != NULL)
        (void)cudaFree(device_sums_b);
    if (device_words != NULL)
        (void)cudaFree(device_words);
    if (device_validation_flags != NULL)
        (void)cudaFree(device_validation_flags);
    if (device_codes != NULL)
        (void)cudaFree(device_codes);
    if (device_payload != NULL)
        (void)cudaFree(device_payload);
    if (device_status != NULL)
        (void)cudaFree(device_status);
    if (event_start != NULL)
        (void)cudaEventDestroy(event_start);
    if (event_stop != NULL)
        (void)cudaEventDestroy(event_stop);
    if (stream != NULL)
        (void)cudaStreamDestroy(stream);
    return result;
}

mco2_q8_status mco2_cuda_q8_compress(const float *values, size_t count,
                                     uint64_t seed, uint64_t tensor_id,
                                     uint64_t invocation_id,
                                     int prescribed_scale_seen,
                                     float prescribed_scale,
                                     const uint32_t *prescribed_words,
                                     uint8_t *payload, float *scale,
                                     int collect_timings,
                                     mco2_cuda_timings *timings)
{
    return mco2_cuda_compress(MCO2_Q8_BITS, values, count, seed, tensor_id,
                              invocation_id, prescribed_scale_seen,
                              prescribed_scale, prescribed_words, 256, 0,
                              payload, scale, collect_timings, timings);
}

struct mco2_cuda_bench_context {
    cudaStream_t stream;
    cudaEvent_t k1_start;
    cudaEvent_t k1_stop;
    cudaEvent_t k2_start;
    cudaEvent_t k2_stop;
    cudaEvent_t k3_start;
    cudaEvent_t k3_stop;
    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;
    float *device_values;
    float *device_scale;
    float *device_k1_scale;
    float *device_max_partials;
    float *device_sums_a;
    float *device_sums_b;
    uint32_t *device_invalid_partials;
    uint32_t *device_words;
    uint32_t *device_validation_flags;
    uint8_t *device_codes;
    uint8_t *device_payload;
    int *device_status;
    uint8_t *host_payload;
    float host_scale;
};

static void destroy_bench_context(mco2_cuda_bench_context *context)
{
    if (context->device_values != NULL)
        (void)cudaFree(context->device_values);
    if (context->device_scale != NULL)
        (void)cudaFree(context->device_scale);
    if (context->device_k1_scale != NULL)
        (void)cudaFree(context->device_k1_scale);
    if (context->device_max_partials != NULL)
        (void)cudaFree(context->device_max_partials);
    if (context->device_invalid_partials != NULL)
        (void)cudaFree(context->device_invalid_partials);
    if (context->device_sums_a != NULL)
        (void)cudaFree(context->device_sums_a);
    if (context->device_sums_b != NULL)
        (void)cudaFree(context->device_sums_b);
    if (context->device_words != NULL)
        (void)cudaFree(context->device_words);
    if (context->device_validation_flags != NULL)
        (void)cudaFree(context->device_validation_flags);
    if (context->device_codes != NULL)
        (void)cudaFree(context->device_codes);
    if (context->device_payload != NULL)
        (void)cudaFree(context->device_payload);
    if (context->device_status != NULL)
        (void)cudaFree(context->device_status);
    if (context->k1_start != NULL)
        (void)cudaEventDestroy(context->k1_start);
    if (context->k1_stop != NULL)
        (void)cudaEventDestroy(context->k1_stop);
    if (context->k2_start != NULL)
        (void)cudaEventDestroy(context->k2_start);
    if (context->k2_stop != NULL)
        (void)cudaEventDestroy(context->k2_stop);
    if (context->k3_start != NULL)
        (void)cudaEventDestroy(context->k3_start);
    if (context->k3_stop != NULL)
        (void)cudaEventDestroy(context->k3_stop);
    if (context->h2d_start != NULL)
        (void)cudaEventDestroy(context->h2d_start);
    if (context->h2d_stop != NULL)
        (void)cudaEventDestroy(context->h2d_stop);
    if (context->d2h_start != NULL)
        (void)cudaEventDestroy(context->d2h_start);
    if (context->d2h_stop != NULL)
        (void)cudaEventDestroy(context->d2h_stop);
    if (context->stream != NULL)
        (void)cudaStreamDestroy(context->stream);
    free(context->host_payload);
}

static mco2_q8_status check_bench_result(float host_scale,
                                         int prescribed_scale_seen,
                                         int host_status,
                                         uint32_t host_validation_flags)
{
    if (host_status != MCO2_Q8_OK &&
        !(prescribed_scale_seen &&
          host_status == MCO2_Q8_ERR_SCALE_OVERFLOW))
        return (mco2_q8_status)host_status;
    if ((host_validation_flags & MCO2_CUDA_INPUT_NONFINITE) != 0)
        return MCO2_Q8_ERR_NONFINITE;
    if (((host_scale == 0.0f) &&
         (host_validation_flags & MCO2_CUDA_INPUT_ANY_NONZERO) != 0) ||
        ((host_scale != 0.0f) &&
         (host_validation_flags & MCO2_CUDA_INPUT_ANY_NONZERO) == 0) ||
        (host_validation_flags & MCO2_CUDA_INPUT_BAD_ZERO_SCALE) != 0)
        return MCO2_Q8_ERR_SCALE;
    return MCO2_Q8_OK;
}

static mco2_q8_status run_bench_pipeline(
    mco2_cuda_bench_context *context, uint8_t bit_width, const float *values,
    size_t count,
    uint64_t seed, uint64_t tensor_id, uint64_t invocation_id,
    int prescribed_scale_seen, int prescribed_words_seen, int block_size,
    int grid_size, mco2_cuda_bench_boundary boundary, int copy_outputs,
    int inspect_result, mco2_bench_sample *sample)
{
    const size_t payload_bytes = bit_width == MCO2_Q4_BITS
                                     ? count / 2 + (count & 1)
                                     : count;
    const uint64_t block_count = count / MCO2_CUDA_REDUCTION_THREADS +
                                 (count % MCO2_CUDA_REDUCTION_THREADS != 0);
    uint64_t padded_count = 1;
    uint32_t host_validation_flags = 0;
    int host_status = MCO2_Q8_OK;
    mco2_rng_stream stream_state;
    const auto wall_start = std::chrono::steady_clock::now();
    cudaError_t error;

    if (mco2_rng_stream_init(&stream_state, seed, tensor_id, invocation_id) !=
        MCO2_Q8_OK)
        return MCO2_Q8_ERR_ID_OVERFLOW;
    while (padded_count < block_count)
        padded_count <<= 1;

    if (boundary == MCO2_CUDA_BENCH_HOST_ORIGIN) {
        error = cudaEventRecord(context->h2d_start, context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        error = cudaMemcpyAsync(context->device_values, values,
                                count * sizeof(float), cudaMemcpyHostToDevice,
                                context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        error = cudaEventRecord(context->h2d_stop, context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
    }

    error = cudaMemsetAsync(context->device_validation_flags, 0,
                            sizeof(uint32_t), context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;

    error = cudaEventRecord(context->k1_start, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    error = (cudaError_t)mco2_cuda_launch_k1(
        context->device_values, count, context->device_max_partials,
        context->device_invalid_partials, context->device_sums_a,
        context->device_sums_b, block_count, padded_count,
        prescribed_scale_seen ? context->device_k1_scale : context->device_scale,
        context->device_status, grid_size, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    error = cudaEventRecord(context->k1_stop, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;

    error = cudaEventRecord(context->k2_start, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    error = (cudaError_t)mco2_cuda_launch_k2(
        bit_width, context->device_values, count, context->device_scale,
        context->device_words, prescribed_words_seen, stream_state,
        context->device_codes, context->device_validation_flags,
        block_size, grid_size, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    error = cudaEventRecord(context->k2_stop, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;

    error = cudaEventRecord(context->k3_start, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    error = (cudaError_t)mco2_cuda_launch_k3(
        bit_width, context->device_codes, count, context->device_payload,
        block_size, grid_size, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    error = cudaEventRecord(context->k3_stop, context->stream);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;

    if (copy_outputs) {
        error = cudaEventRecord(context->d2h_start, context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        if (payload_bytes != 0) {
            error = cudaMemcpyAsync(context->host_payload,
                                    context->device_payload, payload_bytes,
                                    cudaMemcpyDeviceToHost, context->stream);
            if (error != cudaSuccess)
                return MCO2_Q8_ERR_CUDA;
        }
        error = cudaMemcpyAsync(&context->host_scale, context->device_scale,
                                sizeof(float), cudaMemcpyDeviceToHost,
                                context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        error = cudaEventRecord(context->d2h_stop, context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
    }
    if (inspect_result) {
        error = cudaMemcpyAsync(&host_status, context->device_status,
                                sizeof(int), cudaMemcpyDeviceToHost,
                                context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        error = cudaMemcpyAsync(&host_validation_flags,
                                context->device_validation_flags,
                                sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                context->stream);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    if (sample != NULL) {
        const auto wall_stop = std::chrono::steady_clock::now();
        float elapsed = 0.0f;
        sample->wall_ms = std::chrono::duration<double, std::milli>(
                              wall_stop - wall_start)
                              .count();
        error = cudaEventElapsedTime(&elapsed, context->k1_start,
                                     context->k1_stop);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        sample->k1_ms = elapsed;
        error = cudaEventElapsedTime(&elapsed, context->k2_start,
                                     context->k2_stop);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        sample->k2_ms = elapsed;
        error = cudaEventElapsedTime(&elapsed, context->k3_start,
                                     context->k3_stop);
        if (error != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
        sample->k3_ms = elapsed;
        sample->h2d_ms = 0.0;
        sample->d2h_ms = 0.0;
        if (boundary == MCO2_CUDA_BENCH_HOST_ORIGIN) {
            error = cudaEventElapsedTime(&elapsed, context->h2d_start,
                                         context->h2d_stop);
            if (error != cudaSuccess)
                return MCO2_Q8_ERR_CUDA;
            sample->h2d_ms = elapsed;
        }
        if (copy_outputs) {
            error = cudaEventElapsedTime(&elapsed, context->d2h_start,
                                         context->d2h_stop);
            if (error != cudaSuccess)
                return MCO2_Q8_ERR_CUDA;
            sample->d2h_ms = elapsed;
        }
    }

    if (inspect_result)
        return check_bench_result(context->host_scale, prescribed_scale_seen,
                                  host_status, host_validation_flags);
    return MCO2_Q8_OK;
}

/*
 * Captures the resident sequence (1 memset, then K1-K3) once, outside timing,
 * and times each run as one graph launch plus device synchronization. The
 * captured K2 arguments fix the base invocation's RNG stream, so every launch
 * repeats the base workload. After timing, the last launch's outputs must match
 * the uncaptured preflight record byte for byte.
 */
static mco2_q8_status run_bench_graph(
    mco2_cuda_bench_context *context, uint8_t bit_width, size_t count,
    uint64_t seed, uint64_t tensor_id, uint64_t invocation_id,
    int prescribed_scale_seen, int prescribed_words_seen, int block_size,
    int grid_size, uint64_t warmups, uint64_t reps,
    const uint8_t *base_payload, float base_scale, mco2_bench_sample *samples,
    double *capture_ms)
{
    const size_t payload_bytes = bit_width == MCO2_Q4_BITS
                                     ? count / 2 + (count & 1)
                                     : count;
    const uint64_t block_count = count / MCO2_CUDA_REDUCTION_THREADS +
                                 (count % MCO2_CUDA_REDUCTION_THREADS != 0);
    uint64_t padded_count = 1, index;
    uint32_t host_validation_flags = 0;
    int host_status = MCO2_Q8_OK;
    mco2_rng_stream stream_state;
    cudaGraph_t graph = NULL;
    cudaGraphExec_t graph_exec = NULL;
    mco2_q8_status result = MCO2_Q8_ERR_CUDA;
    cudaError_t error, capture_error = cudaSuccess;

    if (mco2_rng_stream_init(&stream_state, seed, tensor_id, invocation_id) !=
        MCO2_Q8_OK)
        return MCO2_Q8_ERR_ID_OVERFLOW;
    while (padded_count < block_count)
        padded_count <<= 1;

    /* Poison the payload left by the preflight so parity proves K3 ran in the graph. */
    if (payload_bytes != 0) {
        for (index = 0; index < payload_bytes; index++)
            context->host_payload[index] = (uint8_t)~base_payload[index];
        if (cudaMemcpy(context->device_payload, context->host_payload,
                       payload_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return MCO2_Q8_ERR_CUDA;
    }

    const auto capture_start = std::chrono::steady_clock::now();
    error = cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeGlobal);
    if (error != cudaSuccess)
        return MCO2_Q8_ERR_CUDA;
    if (count != 0) {
        capture_error = cudaMemsetAsync(context->device_validation_flags, 0,
                                        sizeof(uint32_t), context->stream);
        if (capture_error == cudaSuccess)
            capture_error = (cudaError_t)mco2_cuda_launch_k1(
                context->device_values, count, context->device_max_partials,
                context->device_invalid_partials, context->device_sums_a,
                context->device_sums_b, block_count, padded_count,
                prescribed_scale_seen ? context->device_k1_scale
                                      : context->device_scale,
                context->device_status, grid_size, context->stream);
        if (capture_error == cudaSuccess)
            capture_error = (cudaError_t)mco2_cuda_launch_k2(
                bit_width, context->device_values, count, context->device_scale,
                context->device_words, prescribed_words_seen, stream_state,
                context->device_codes, context->device_validation_flags,
                block_size, grid_size, context->stream);
        if (capture_error == cudaSuccess)
            capture_error = (cudaError_t)mco2_cuda_launch_k3(
                bit_width, context->device_codes, count,
                context->device_payload, block_size, grid_size,
                context->stream);
    }
    /* End the capture even after a failed launch so the stream leaves capture mode. */
    error = cudaStreamEndCapture(context->stream, &graph);
    if (capture_error != cudaSuccess || error != cudaSuccess)
        goto done;
    error = cudaGraphInstantiate(&graph_exec, graph, 0);
    if (error != cudaSuccess)
        goto done;
    if (capture_ms != NULL)
        *capture_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - capture_start)
                          .count();

    for (index = 0; index < warmups + reps; index++) {
        mco2_bench_sample *sample = index < warmups ? NULL : &samples[index - warmups];
        const auto wall_start = std::chrono::steady_clock::now();
        error = cudaGraphLaunch(graph_exec, context->stream);
        if (error == cudaSuccess)
            error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            goto done;
        if (sample != NULL) {
            const auto wall_stop = std::chrono::steady_clock::now();
            sample->wall_ms = std::chrono::duration<double, std::milli>(
                                  wall_stop - wall_start)
                                  .count();
            sample->k1_ms = 0.0;
            sample->k2_ms = 0.0;
            sample->k3_ms = 0.0;
            sample->h2d_ms = 0.0;
            sample->d2h_ms = 0.0;
        }
    }

    if (count != 0) {
        if (payload_bytes != 0)
            error = cudaMemcpyAsync(context->host_payload,
                                    context->device_payload, payload_bytes,
                                    cudaMemcpyDeviceToHost, context->stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(&context->host_scale, context->device_scale,
                                    sizeof(float), cudaMemcpyDeviceToHost,
                                    context->stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(&host_status, context->device_status,
                                    sizeof(int), cudaMemcpyDeviceToHost,
                                    context->stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(&host_validation_flags,
                                    context->device_validation_flags,
                                    sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                    context->stream);
        if (error == cudaSuccess)
            error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            goto done;
        result = check_bench_result(context->host_scale, prescribed_scale_seen,
                                    host_status, host_validation_flags);
        if (result != MCO2_Q8_OK)
            goto done;
        result = MCO2_Q8_ERR_CUDA;
        if (memcmp(&context->host_scale, &base_scale, sizeof(float)) != 0 ||
            (payload_bytes != 0 &&
             memcmp(context->host_payload, base_payload, payload_bytes) != 0))
            goto done;
    }
    result = MCO2_Q8_OK;

done:
    if (graph_exec != NULL)
        (void)cudaGraphExecDestroy(graph_exec);
    if (graph != NULL)
        (void)cudaGraphDestroy(graph);
    return result;
}

mco2_q8_status mco2_cuda_bench(
    uint8_t bit_width, const float *values, size_t count, uint64_t seed,
    uint64_t tensor_id, uint64_t base_invocation_id,
    int prescribed_scale_seen, float prescribed_scale,
    const uint32_t *prescribed_words, int block_size, int grid_size,
    mco2_cuda_bench_boundary boundary, uint64_t warmups, uint64_t reps,
    uint8_t *base_payload, float *base_scale, mco2_bench_sample *samples,
    double *capture_ms)
{
    mco2_cuda_bench_context context = {};
    mco2_q8_status result = MCO2_Q8_ERR_CUDA;
    cudaError_t error;
    int device_count = 0;
    uint64_t total_runs, block_count, padded_count = 1, index;
    size_t payload_bytes;

    if (capture_ms != NULL)
        *capture_ms = 0.0;
    if (bit_width != MCO2_Q4_BITS && bit_width != MCO2_Q8_BITS)
        return MCO2_Q8_ERR_BIT_WIDTH;
    if ((count != 0 && (values == NULL || base_payload == NULL)) ||
        base_scale == NULL || samples == NULL || reps == 0 ||
        count > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(uint32_t))
        return MCO2_Q8_ERR_ARGUMENT;
    if (block_size <= 0 || block_size > MCO2_CUDA_MAX_BLOCK_SIZE ||
        grid_size < 0 || grid_size > MCO2_CUDA_MAX_GRID_SIZE ||
        (boundary != MCO2_CUDA_BENCH_RESIDENT &&
         boundary != MCO2_CUDA_BENCH_HOST_ORIGIN &&
         boundary != MCO2_CUDA_BENCH_RESIDENT_GRAPH))
        return MCO2_Q8_ERR_ARGUMENT;
    if (prescribed_scale_seen &&
        (!std::isfinite(prescribed_scale) || prescribed_scale < 0.0f))
        return MCO2_Q8_ERR_SCALE;
    if (tensor_id > UINT32_MAX || base_invocation_id > UINT32_MAX ||
        warmups > UINT64_MAX - reps ||
        (total_runs = warmups + reps) == 0 ||
        total_runs - 1 > (uint64_t)UINT32_MAX - base_invocation_id)
        return MCO2_Q8_ERR_ID_OVERFLOW;
    if (prescribed_scale_seen && count == 0 && prescribed_scale != 0.0f)
        return MCO2_Q8_ERR_SCALE;

    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        (void)cudaGetLastError();
        return MCO2_Q8_ERR_CUDA;
    }
    error = cudaStreamCreateWithFlags(&context.stream, cudaStreamNonBlocking);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.k1_start);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.k1_stop);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.k2_start);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.k2_stop);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.k3_start);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.k3_stop);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.h2d_start);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.h2d_stop);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.d2h_start);
    if (error != cudaSuccess)
        goto done;
    error = cudaEventCreate(&context.d2h_stop);
    if (error != cudaSuccess)
        goto done;

    payload_bytes = bit_width == MCO2_Q4_BITS ? count / 2 + (count & 1) : count;
    context.host_payload = (uint8_t *)malloc(payload_bytes == 0 ? 1 : payload_bytes);
    if (context.host_payload == NULL) {
        result = MCO2_Q8_ERR_MEMORY;
        goto done;
    }
    if (count != 0) {
#define BENCH_ALLOCATE(pointer, bytes)                                                    \
        do {                                                                               \
            error = cudaMalloc(reinterpret_cast<void **>(&(pointer)), (bytes));             \
            if (error != cudaSuccess)                                                      \
                goto done;                                                                 \
        } while (0)
        BENCH_ALLOCATE(context.device_values, count * sizeof(float));
        BENCH_ALLOCATE(context.device_scale, sizeof(float));
        BENCH_ALLOCATE(context.device_status, sizeof(int));
        BENCH_ALLOCATE(context.device_validation_flags, sizeof(uint32_t));
        BENCH_ALLOCATE(context.device_codes, count * sizeof(uint8_t));
        BENCH_ALLOCATE(context.device_payload,
                       (payload_bytes == 0 ? 1 : payload_bytes) * sizeof(uint8_t));
        if (prescribed_scale_seen)
            BENCH_ALLOCATE(context.device_k1_scale, sizeof(float));
        if (prescribed_words != NULL)
            BENCH_ALLOCATE(context.device_words, count * sizeof(uint32_t));
        block_count = count / MCO2_CUDA_REDUCTION_THREADS +
                      (count % MCO2_CUDA_REDUCTION_THREADS != 0);
        while (padded_count < block_count) {
            if (padded_count > UINT64_MAX / 2) {
                result = MCO2_Q8_ERR_COUNT;
                goto done;
            }
            padded_count <<= 1;
        }
        BENCH_ALLOCATE(context.device_max_partials,
                       block_count * sizeof(float));
        BENCH_ALLOCATE(context.device_invalid_partials,
                       block_count * sizeof(uint32_t));
        BENCH_ALLOCATE(context.device_sums_a, padded_count * sizeof(float));
        BENCH_ALLOCATE(context.device_sums_b, padded_count * sizeof(float));
#undef BENCH_ALLOCATE
        if (prescribed_scale_seen) {
            error = cudaMemcpyAsync(context.device_scale, &prescribed_scale,
                                    sizeof(float), cudaMemcpyHostToDevice,
                                    context.stream);
            if (error != cudaSuccess)
                goto done;
        }
        if (prescribed_words != NULL) {
            error = cudaMemcpyAsync(context.device_words, prescribed_words,
                                    count * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice, context.stream);
            if (error != cudaSuccess)
                goto done;
        }
        if (boundary == MCO2_CUDA_BENCH_RESIDENT ||
            boundary == MCO2_CUDA_BENCH_RESIDENT_GRAPH) {
            error = cudaMemcpyAsync(context.device_values, values,
                                    count * sizeof(float),
                                    cudaMemcpyHostToDevice, context.stream);
            if (error != cudaSuccess)
                goto done;
        }
        error = cudaStreamSynchronize(context.stream);
        if (error != cudaSuccess)
            goto done;
    }

    if (count == 0) {
        *base_scale = 0.0f;
        if (boundary == MCO2_CUDA_BENCH_RESIDENT_GRAPH) {
            result = run_bench_graph(
                &context, bit_width, count, seed, tensor_id, base_invocation_id,
                prescribed_scale_seen, prescribed_words != NULL, block_size,
                grid_size, warmups, reps, base_payload, *base_scale, samples,
                capture_ms);
            goto done;
        }
        for (index = 0; index < warmups + reps; index++) {
            mco2_bench_sample *sample = index < warmups ? NULL : &samples[index - warmups];
            const auto start = std::chrono::steady_clock::now();
            error = cudaDeviceSynchronize();
            if (error != cudaSuccess) {
                result = MCO2_Q8_ERR_CUDA;
                goto done;
            }
            if (sample != NULL) {
                const auto stop = std::chrono::steady_clock::now();
                sample->wall_ms = std::chrono::duration<double, std::milli>(stop - start).count();
                sample->k1_ms = 0.0;
                sample->k2_ms = 0.0;
                sample->k3_ms = 0.0;
                sample->h2d_ms = 0.0;
                sample->d2h_ms = 0.0;
            }
        }
        result = MCO2_Q8_OK;
        goto done;
    }

    result = run_bench_pipeline(
        &context, bit_width, values, count, seed, tensor_id, base_invocation_id,
        prescribed_scale_seen, prescribed_words != NULL, block_size, grid_size,
        boundary == MCO2_CUDA_BENCH_HOST_ORIGIN ? MCO2_CUDA_BENCH_HOST_ORIGIN : MCO2_CUDA_BENCH_RESIDENT,
        1, 1, NULL);
    if (result != MCO2_Q8_OK)
        goto done;
    if (payload_bytes != 0)
        memcpy(base_payload, context.host_payload, payload_bytes);
    *base_scale = context.host_scale;

    if (boundary == MCO2_CUDA_BENCH_RESIDENT_GRAPH) {
        result = run_bench_graph(
            &context, bit_width, count, seed, tensor_id, base_invocation_id,
            prescribed_scale_seen, prescribed_words != NULL, block_size,
            grid_size, warmups, reps, base_payload, *base_scale, samples,
            capture_ms);
        goto done;
    }

    for (index = 0; index < total_runs; index++) {
        mco2_bench_sample *sample = index < warmups ? NULL : &samples[index - warmups];
        const uint64_t run_offset = index < warmups
            ? reps + index
            : index - warmups;
        const uint64_t invocation_id = base_invocation_id + run_offset;
        result = run_bench_pipeline(
            &context, bit_width, values, count, seed, tensor_id, invocation_id,
            prescribed_scale_seen, prescribed_words != NULL, block_size,
            grid_size, boundary,
            boundary == MCO2_CUDA_BENCH_HOST_ORIGIN, 0, sample);
        if (result != MCO2_Q8_OK)
            goto done;
    }
    result = MCO2_Q8_OK;

done:
    destroy_bench_context(&context);
    return result;
}
