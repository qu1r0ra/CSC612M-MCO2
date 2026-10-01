#include "rng_cpu.h"

void sq_rng_words_cpu(const sq_rng_stream *s, uint64_t n, uint32_t *out) {
  philox4x32_key_t key = sq_philox_key(s);
  uint64_t i = 0;

  while (i < n) {
    philox4x32_ctr_t r = philox4x32_R(
        SQ_PHILOX_ROUNDS, sq_philox_ctr(s, SQ_PHILOX_WORD_GROUP(i)), key);
    out[i] = r.v[SQ_PHILOX_RESULT_0];
    i++;
    if (i < n) {
      out[i] = r.v[SQ_PHILOX_RESULT_1];
      i++;
    }
    if (i < n) {
      out[i] = r.v[SQ_PHILOX_RESULT_2];
      i++;
    }
    if (i < n) {
      out[i] = r.v[SQ_PHILOX_RESULT_3];
      i++;
    }
  }
}
