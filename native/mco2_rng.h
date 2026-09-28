#ifndef MCO2_RNG_H
#define MCO2_RNG_H

#include <stdint.h>

#include <Random123/philox.h>

#ifdef __CUDACC__
#define MCO2_HD static __host__ __device__ __forceinline__
#else
#define MCO2_HD static __inline
#endif

#define MCO2_PHILOX_ROUNDS 10

enum {
    MCO2_OK = 0,
    MCO2_ERR_ID_OVERFLOW = 1
};

/* One random stream per (seed, tensor, invocation). */
typedef struct {
    uint64_t seed;
    uint32_t tensor_id;
    uint32_t invocation_id;
} mco2_rng_stream;

/*
 * Identifiers arrive as 64-bit values from callers and the CLI. Values that do
 * not fit the 32-bit counter words are rejected instead of truncated, so two
 * logical streams can never alias.
 */
MCO2_HD int mco2_rng_stream_init(mco2_rng_stream *s, uint64_t seed,
                                 uint64_t tensor_id, uint64_t invocation_id)
{
    if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX)
        return MCO2_ERR_ID_OVERFLOW;
    s->seed = seed;
    s->tensor_id = (uint32_t)tensor_id;
    s->invocation_id = (uint32_t)invocation_id;
    return MCO2_OK;
}

/* key = {seed lo, seed hi} */
MCO2_HD philox4x32_key_t mco2_philox_key(const mco2_rng_stream *s)
{
    philox4x32_key_t k;
    k.v[0] = (uint32_t)s->seed;
    k.v[1] = (uint32_t)(s->seed >> 32);
    return k;
}

/* counter = {group lo, group hi, tensor, invocation}; group = floor(i/4) */
MCO2_HD philox4x32_ctr_t mco2_philox_ctr(const mco2_rng_stream *s, uint64_t group)
{
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
MCO2_HD uint64_t mco2_bernoulli_threshold(float p)
{
    return (uint64_t)(p * 4294967296.0f);
}

MCO2_HD int mco2_bernoulli(uint32_t word, float p)
{
    return (uint64_t)word < mco2_bernoulli_threshold(p);
}

#endif
