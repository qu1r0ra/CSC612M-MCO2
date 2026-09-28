#ifndef SQ_RNG_H
#define SQ_RNG_H

#include <stdint.h>

#include <Random123/philox.h>

#ifdef __CUDACC__
#define SQ_HD static __host__ __device__ __forceinline__
#else
#define SQ_HD static __inline
#endif

#define SQ_PHILOX_ROUNDS 10

enum { SQ_RNG_OK = 0, SQ_RNG_ERR_ID_OVERFLOW = 1 };

/* One random stream per (seed, tensor, invocation). */
typedef struct {
  uint64_t seed;
  uint32_t tensor_id;
  uint32_t invocation_id;
} sq_rng_stream;

/*
 * Identifiers arrive as 64-bit values from callers and the CLI. Values that do
 * not fit the 32-bit counter words are rejected instead of truncated, so two
 * logical streams can never alias.
 */
SQ_HD int sq_rng_stream_init(sq_rng_stream *s, uint64_t seed,
                             uint64_t tensor_id, uint64_t invocation_id) {
  if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX) {
    return SQ_RNG_ERR_ID_OVERFLOW;
  }
  s->seed = seed;
  s->tensor_id = (uint32_t)tensor_id;
  s->invocation_id = (uint32_t)invocation_id;
  return SQ_RNG_OK;
}

/* key = {seed lo, seed hi} */
SQ_HD philox4x32_key_t sq_philox_key(const sq_rng_stream *s) {
  philox4x32_key_t k;
  k.v[0] = (uint32_t)s->seed;
  k.v[1] = (uint32_t)(s->seed >> 32);
  return k;
}

/* counter = {group lo, group hi, tensor, invocation}; group = floor(i/4) */
SQ_HD philox4x32_ctr_t sq_philox_ctr(const sq_rng_stream *s, uint64_t group) {
  philox4x32_ctr_t c;
  c.v[0] = (uint32_t)group;
  c.v[1] = (uint32_t)(group >> 32);
  c.v[2] = s->tensor_id;
  c.v[3] = s->invocation_id;
  return c;
}

/*
 * floor(p * 2^32) for p in [0, 1]. Scaling by 2^32 is exact in FP32, and the
 * result can be 2^32 at p = 1, so the threshold needs 64 bits.
 */
SQ_HD uint64_t sq_bernoulli_threshold(float p) {
  return (uint64_t)(p * 4294967296.0f);
}

SQ_HD int sq_bernoulli(uint32_t word, float p) {
  return (uint64_t)word < sq_bernoulli_threshold(p);
}

#endif
