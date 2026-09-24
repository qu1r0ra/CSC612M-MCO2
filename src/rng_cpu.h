#ifndef MCO2_RNG_CPU_H
#define MCO2_RNG_CPU_H

#include "mco2_rng.h"

philox4x32_ctr_t mco2_philox_raw_cpu(philox4x32_ctr_t ctr, philox4x32_key_t key);

/* out[i] = word for element i under the logical mapping. */
void mco2_rng_words_cpu(const mco2_rng_stream *s, uint64_t n, uint32_t *out);

void mco2_bernoulli_cpu(const uint32_t *words, const float *p, uint64_t n,
                        uint8_t *out);

#endif
