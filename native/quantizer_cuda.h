#ifndef SQ_QUANTIZER_CUDA_H
#define SQ_QUANTIZER_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "sq_rng.h"

#define SQ_CUDA_MAX_BLOCK_SIZE 1024
#define SQ_CUDA_MAX_GRID_SIZE 65535

typedef struct {
    float k1_ms;
    float k2_ms;
    float k3_ms;
    float h2d_ms;
    float d2h_ms;
} sq_cuda_timings;

typedef struct {
    double wall_ms;
    double k1_ms;
    double k2_ms;
    double k3_ms;
    double h2d_ms;
    double d2h_ms;
    double cpu_ms;
} sq_bench_sample;

typedef enum {
    SQ_CUDA_BENCH_RESIDENT = 0,
    SQ_CUDA_BENCH_HOST_ORIGIN = 1,
    SQ_CUDA_BENCH_RESIDENT_GRAPH = 2,
    SQ_CUDA_BENCH_GPU_ORIGIN = 3
} sq_cuda_bench_boundary;

/* Host buffers that a timed transfer touches: malloc or cudaHostAlloc. */
typedef enum {
    SQ_CUDA_TRANSFER_PAGEABLE = 0,
    SQ_CUDA_TRANSFER_PINNED = 1
} sq_cuda_transfer_policy;

/* Device-resident input for the CPU GPU-origin path, which downloads the input
   to a host landing buffer before each CPU compression. */
typedef struct sq_cuda_staging sq_cuda_staging;

#ifdef __cplusplus
extern "C" {
#endif

/* K1 variants (issue #23). Both write the same scale bit for bit. The
   optimized one needs a 16-byte-aligned input (cudaMalloc pointers are) and
   returns cudaErrorMisalignedAddress otherwise. */
enum { SQ_CUDA_K1_REFERENCE = 0, SQ_CUDA_K1_OPTIMIZED = 1 };

/* Selects the K1 variant for every later launch in this process; the CLI
   calls it once before any CUDA work. Returns 0 or cudaErrorInvalidValue. */
int sq_cuda_select_k1(int variant);

/* Stage launchers accept device pointers and an opaque CUDA stream handle. */
int sq_cuda_launch_k1(const float *device_values, uint64_t count,
                      float *device_max_partials,
                      uint32_t *device_invalid_partials,
                      float *device_sums_a, float *device_sums_b,
                      uint64_t block_count, uint64_t padded_count,
                      float *device_scale, int *device_status,
                      int grid_size, void *stream);

int sq_cuda_launch_q8_k1(const float *device_values, uint64_t count,
                         float *device_max_partials,
                         uint32_t *device_invalid_partials,
                         float *device_sums_a, float *device_sums_b,
                         uint64_t block_count, uint64_t padded_count,
                         float *device_scale, int *device_status,
                         void *stream);

int sq_cuda_launch_k2(uint8_t bit_width, const float *device_values,
                      uint64_t count, const float *device_scale,
                      const uint32_t *device_words, int prescribed_words,
                      sq_rng_stream stream_state,
                      uint8_t *device_codes,
                      uint32_t *device_validation_flags,
                      int block_size, int grid_size, void *stream);

int sq_cuda_launch_q8_k2(const float *device_values, uint64_t count,
                         const float *device_scale,
                         const uint32_t *device_words, int prescribed_words,
                         sq_rng_stream stream_state,
                         uint8_t *device_codes,
                         uint32_t *device_validation_flags,
                         int block_size, int grid_size, void *stream);

int sq_cuda_launch_k3(uint8_t bit_width, const uint8_t *device_codes,
                      uint64_t count, uint8_t *device_payload,
                      int block_size, int grid_size, void *stream);

int sq_cuda_launch_q4_k3(const uint8_t *device_codes, uint64_t count,
                         uint8_t *device_payload, int block_size,
                         int grid_size, void *stream);

int sq_cuda_launch_q8_k3(const uint8_t *device_codes, uint64_t count,
                         uint8_t *device_payload, int block_size,
                         int grid_size, void *stream);

/* Host-level C entry points for compression pipelines. */
sq_status sq_cuda_compress(uint8_t bit_width, const float *values,
                           size_t count, uint64_t seed,
                           uint64_t tensor_id, uint64_t invocation_id,
                           int prescribed_scale_seen,
                           float prescribed_scale,
                           const uint32_t *prescribed_words,
                           int block_size, int grid_size,
                           uint8_t *payload, float *scale,
                           int collect_timings,
                           sq_cuda_timings *timings);

sq_status sq_cuda_q8_compress(const float *values, size_t count,
                              uint64_t seed, uint64_t tensor_id,
                              uint64_t invocation_id,
                              int prescribed_scale_seen,
                              float prescribed_scale,
                              const uint32_t *prescribed_words,
                              uint8_t *payload, float *scale,
                              int collect_timings,
                              sq_cuda_timings *timings);

/* Runs the base record preflight, warmups, and measured repetitions with one
   reusable CUDA allocation/event context. The payload and scale outputs are
   from the untimed base-configuration run. */
sq_status sq_cuda_bench(
    uint8_t bit_width, const float *values, size_t count, uint64_t seed,
    uint64_t tensor_id, uint64_t base_invocation_id,
    int prescribed_scale_seen, float prescribed_scale,
    const uint32_t *prescribed_words, int block_size, int grid_size,
    sq_cuda_bench_boundary boundary,
    sq_cuda_transfer_policy transfer_policy, uint64_t warmups, uint64_t reps,
    uint8_t *base_payload, float *base_scale, sq_bench_sample *samples,
    double *capture_ms);

/* Uploads `values` to the device and allocates the landing buffer, outside
   timing. The landing buffer holds no input until the first download. */
sq_status sq_cuda_staging_create(const float *values, size_t count,
                                 sq_cuda_transfer_policy transfer_policy,
                                 sq_cuda_staging **staging);

/* Copies the device input into the landing buffer and waits for it. The copy
   time comes from CUDA events around the copy. */
sq_status sq_cuda_staging_download(sq_cuda_staging *staging,
                                   const float **landing,
                                   double *d2h_ms);

void sq_cuda_staging_destroy(sq_cuda_staging *staging);

#ifdef __cplusplus
}
#endif

#endif
