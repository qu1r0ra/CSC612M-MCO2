#ifndef MCO2_RNG_CUDA_H
#define MCO2_RNG_CUDA_H

#include "mco2_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Each launch function takes host pointers and returns 0 or a cudaError_t. */

int mco2_cuda_device_info(char *name, int name_len, int *major, int *minor);

int mco2_philox_raw_cuda(philox4x32_ctr_t ctr, philox4x32_key_t key,
                         philox4x32_ctr_t *out);

/* Same contract as mco2_rng_words_cpu, one thread per element. */
int mco2_rng_words_cuda(const mco2_rng_stream *s, uint64_t n, uint32_t *out,
                        int block_size, int grid_size);

int mco2_bernoulli_cuda(const uint32_t *words, const float *p, uint64_t n,
                        uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif
