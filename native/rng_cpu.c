#include "rng_cpu.h"

philox4x32_ctr_t sq_philox_raw_cpu(philox4x32_ctr_t ctr, philox4x32_key_t key) {
  return philox4x32_R(SQ_PHILOX_ROUNDS, ctr, key);
}

void sq_rng_words_cpu(const sq_rng_stream *s, uint64_t n, uint32_t *out) {
  philox4x32_key_t key = sq_philox_key(s);
  uint64_t i;

  for (i = 0; i < n; i++) {
    philox4x32_ctr_t r =
        philox4x32_R(SQ_PHILOX_ROUNDS, sq_philox_ctr(s, i / 4), key);
    out[i] = r.v[i % 4];
  }
}

void sq_bernoulli_cpu(const uint32_t *words, const float *p, uint64_t n,
                      uint8_t *out) {
  uint64_t i;

  for (i = 0; i < n; i++) {
    out[i] = (uint8_t)sq_bernoulli(words[i], p[i]);
  }
}
