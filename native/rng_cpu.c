#include "rng_cpu.h"

philox4x32_ctr_t mco2_philox_raw_cpu(philox4x32_ctr_t ctr, philox4x32_key_t key)
{
    return philox4x32_R(MCO2_PHILOX_ROUNDS, ctr, key);
}

void mco2_rng_words_cpu(const mco2_rng_stream *s, uint64_t n, uint32_t *out)
{
    philox4x32_key_t key = mco2_philox_key(s);
    uint64_t i;

    for (i = 0; i < n; i++) {
        philox4x32_ctr_t r = philox4x32_R(MCO2_PHILOX_ROUNDS,
                                          mco2_philox_ctr(s, i / 4), key);
        out[i] = r.v[i % 4];
    }
}

void mco2_bernoulli_cpu(const uint32_t *words, const float *p, uint64_t n,
                        uint8_t *out)
{
    uint64_t i;

    for (i = 0; i < n; i++)
        out[i] = (uint8_t)mco2_bernoulli(words[i], p[i]);
}
