#ifndef SQ_QUANTIZER_CUDA_H
#define SQ_QUANTIZER_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "bench_types.h"
#include "codec.h"
#include "cuda_limits.h"
#include "sq_rng.h"

/* Grid for a launch over `work` items. Kernels are grid-stride loops, so
   capping the grid never skips elements. */
static inline int sq_cuda_automatic_grid(uint64_t work, int block_size) {
  const uint64_t blocks =
      work / (uint64_t)block_size + (work % (uint64_t)block_size != 0);
  return (int)(blocks < SQ_CUDA_MAX_GRID_SIZE ? blocks : SQ_CUDA_MAX_GRID_SIZE);
}

typedef struct {
  float k1_ms;
  float k2_ms;
  float k3_ms;
  float h2d_ms;
  float d2h_ms;
} sq_cuda_timings;

/* The driver's boundaries and transfer policies; the values are the neutral
   ones the CLI parses (bench_types.h). */
typedef enum {
  SQ_CUDA_BENCH_RESIDENT = SQ_BOUNDARY_RESIDENT,
  SQ_CUDA_BENCH_HOST_ORIGIN = SQ_BOUNDARY_HOST_ORIGIN,
  SQ_CUDA_BENCH_RESIDENT_GRAPH = SQ_BOUNDARY_RESIDENT_GRAPH,
  SQ_CUDA_BENCH_GPU_ORIGIN = SQ_BOUNDARY_GPU_ORIGIN
} sq_cuda_bench_boundary;

typedef enum {
  SQ_CUDA_TRANSFER_PAGEABLE = SQ_TRANSFER_PAGEABLE,
  SQ_CUDA_TRANSFER_PINNED = SQ_TRANSFER_PINNED
} sq_cuda_transfer_policy;

/* Device-resident input for the CPU GPU-origin path, which downloads the input
   to a host landing buffer before each CPU compression. */
typedef struct sq_cuda_staging sq_cuda_staging;

#ifdef __cplusplus
extern "C" {
#endif

/* The optimized K1 variant needs a 16-byte-aligned input (cudaMalloc pointers
   are) and returns cudaErrorMisalignedAddress otherwise. */

/* Selects the K1 variant for every later launch in this process; the CLI
   calls it once before any CUDA work. Returns 0 or cudaErrorInvalidValue. */
int sq_cuda_select_k1(int variant);

/* Stage launchers accept device pointers and an opaque CUDA stream handle. */
int sq_cuda_launch_k1(const float *device_values, uint64_t count,
                      float *device_max_partials,
                      uint32_t *device_invalid_partials, float *device_sums_a,
                      float *device_sums_b, uint64_t block_count,
                      uint64_t padded_count, float *device_scale,
                      int *device_status, int grid_size, void *stream);

int sq_cuda_launch_k2(uint8_t bit_width, const float *device_values,
                      uint64_t count, const float *device_scale,
                      const uint32_t *device_words, int prescribed_words,
                      sq_rng_stream stream_state, uint8_t *device_codes,
                      uint32_t *device_validation_flags, int block_size,
                      int grid_size, void *stream);

int sq_cuda_launch_k3(uint8_t bit_width, const uint8_t *device_codes,
                      uint64_t count, uint8_t *device_payload, int block_size,
                      int grid_size, void *stream);

/* Host-level C entry points for compression pipelines. */
sq_status sq_cuda_compress(uint8_t bit_width, const float *values, size_t count,
                           uint64_t seed, uint64_t tensor_id,
                           uint64_t invocation_id, int prescribed_scale_seen,
                           float prescribed_scale,
                           const uint32_t *prescribed_words, int block_size,
                           int grid_size, uint8_t *payload, float *scale,
                           int collect_timings, sq_cuda_timings *timings);

/* Runs the base record preflight, warmups, and measured repetitions with one
   reusable CUDA allocation/event context. The payload and scale outputs are
   from the untimed base-configuration run. */
sq_status sq_cuda_bench(uint8_t bit_width, const float *values, size_t count,
                        uint64_t seed, uint64_t tensor_id,
                        uint64_t base_invocation_id, int prescribed_scale_seen,
                        float prescribed_scale,
                        const uint32_t *prescribed_words, int block_size,
                        int grid_size, sq_cuda_bench_boundary boundary,
                        sq_cuda_transfer_policy transfer_policy,
                        uint64_t warmups, uint64_t reps, uint8_t *base_payload,
                        float *base_scale, sq_bench_sample *samples,
                        double *capture_ms);

/* Uploads `values` to the device and allocates the landing buffer, outside
   timing. The landing buffer holds no input until the first download. */
sq_status sq_cuda_staging_create(const float *values, size_t count,
                                 sq_cuda_transfer_policy transfer_policy,
                                 sq_cuda_staging **staging);

/* Copies the device input into the landing buffer and waits for it. The copy
   time comes from CUDA events around the copy. */
sq_status sq_cuda_staging_download(sq_cuda_staging *staging,
                                   const float **landing, double *d2h_ms);

void sq_cuda_staging_destroy(sq_cuda_staging *staging);

#ifdef __cplusplus
}
#endif

#endif
