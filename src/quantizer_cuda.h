#ifndef MCO2_QUANTIZER_CUDA_H
#define MCO2_QUANTIZER_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "mco2_rng.h"

#define MCO2_CUDA_MAX_BLOCK_SIZE 1024
#define MCO2_CUDA_MAX_GRID_SIZE 65535

typedef struct {
    float k1_ms;
    float k2_ms;
    float k3_ms;
    float h2d_ms;
    float d2h_ms;
} mco2_cuda_timings;

#ifdef __cplusplus
extern "C" {
#endif

/* Stage launchers accept device pointers and an opaque CUDA stream handle. */
int mco2_cuda_launch_k1(const float *device_values, uint64_t count,
                        float *device_max_partials,
                        uint32_t *device_invalid_partials,
                        float *device_sums_a, float *device_sums_b,
                        uint64_t block_count, uint64_t padded_count,
                        float *device_scale, int *device_status,
                        int grid_size, void *stream);

int mco2_cuda_launch_q8_k1(const float *device_values, uint64_t count,
                           float *device_max_partials,
                           uint32_t *device_invalid_partials,
                           float *device_sums_a, float *device_sums_b,
                           uint64_t block_count, uint64_t padded_count,
                           float *device_scale, int *device_status,
                           void *stream);

int mco2_cuda_launch_k2(uint8_t bit_width, const float *device_values,
                        uint64_t count, const float *device_scale,
                        const uint32_t *device_words, int prescribed_words,
                        mco2_rng_stream stream_state,
                        uint8_t *device_codes,
                        uint32_t *device_validation_flags,
                        int block_size, int grid_size, void *stream);

int mco2_cuda_launch_q8_k2(const float *device_values, uint64_t count,
                           const float *device_scale,
                           const uint32_t *device_words, int prescribed_words,
                           mco2_rng_stream stream_state,
                           uint8_t *device_codes,
                           uint32_t *device_validation_flags,
                           int block_size, int grid_size, void *stream);

int mco2_cuda_launch_k3(uint8_t bit_width, const uint8_t *device_codes,
                        uint64_t count, uint8_t *device_payload,
                        int block_size, int grid_size, void *stream);

int mco2_cuda_launch_q4_k3(const uint8_t *device_codes, uint64_t count,
                           uint8_t *device_payload, int block_size,
                           int grid_size, void *stream);

int mco2_cuda_launch_q8_k3(const uint8_t *device_codes, uint64_t count,
                           uint8_t *device_payload, int block_size,
                           int grid_size, void *stream);

/* Host-level C entry points for compression pipelines. */
mco2_q8_status mco2_cuda_compress(uint8_t bit_width, const float *values,
                                  size_t count, uint64_t seed,
                                  uint64_t tensor_id, uint64_t invocation_id,
                                  int prescribed_scale_seen,
                                  float prescribed_scale,
                                  const uint32_t *prescribed_words,
                                  int block_size, int grid_size,
                                  uint8_t *payload, float *scale,
                                  int collect_timings,
                                  mco2_cuda_timings *timings);

mco2_q8_status mco2_cuda_q8_compress(const float *values, size_t count,
                                     uint64_t seed, uint64_t tensor_id,
                                     uint64_t invocation_id,
                                     int prescribed_scale_seen,
                                     float prescribed_scale,
                                     const uint32_t *prescribed_words,
                                     uint8_t *payload, float *scale,
                                     int collect_timings,
                                     mco2_cuda_timings *timings);

#ifdef __cplusplus
}
#endif

#endif
