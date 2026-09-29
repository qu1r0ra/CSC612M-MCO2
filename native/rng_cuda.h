#ifndef SQ_RNG_CUDA_H
#define SQ_RNG_CUDA_H

#include "sq_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Test adapter: exercises the Philox and Bernoulli device code against the CPU
   reference. Not linked into the stoquant binary. Each launch function takes
   host pointers and returns 0 or a cudaError_t. */

int sq_cuda_device_info(char *name, int name_len, int *major, int *minor);

int sq_philox_raw_cuda(philox4x32_ctr_t ctr, philox4x32_key_t key,
                       philox4x32_ctr_t *out);

/* Same contract as sq_rng_words_cpu; grid-stride, so any grid covers n. */
int sq_rng_words_cuda(const sq_rng_stream *s, uint64_t n, uint32_t *out,
                      int block_size, int grid_size);

int sq_bernoulli_cuda(const uint32_t *words, const float *p, uint64_t n,
                      uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif
