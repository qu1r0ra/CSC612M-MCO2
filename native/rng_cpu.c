#include "rng_cpu.h"

void sq_rng_words_cpu(const sq_rng_stream *s, uint64_t n, uint32_t *out) {
  philox4x32_key_t key = sq_philox_key(s);
  uint64_t i = 0;

  while (i < n) {
    philox4x32_ctr_t r =
        philox4x32_R(SQ_PHILOX_ROUNDS, sq_philox_ctr(s, i / 4), key);
    out[i] = r.v[0];
    i++;
    if (i < n) {
      out[i] = r.v[1];
      i++;
    }
    if (i < n) {
      out[i] = r.v[2];
      i++;
    }
    if (i < n) {
      out[i] = r.v[3];
      i++;
    }
  }
}
