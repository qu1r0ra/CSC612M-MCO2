#ifndef SQ_QUANTIZER_AVX2_H
#define SQ_QUANTIZER_AVX2_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "sq_rng.h"

/*
 * Multithreaded AVX2 CPU comparator (issue #22). Each stage returns the same
 * status and writes the same bytes as its scalar counterpart in quantizer.c and
 * rng_cpu.c for every thread count. This file is compiled with /arch:AVX2, so
 * callers must confirm AVX2 support before calling into it.
 */

/* Team size a parallel region actually gets for the requested thread count. */
int sq_avx2_team_size(int threads);
int sq_avx2_is_supported(void);

sq_status sq_avx2_compute_scale_with_workspace(const float *values,
                                               size_t count, float *scale,
                                               float *partials,
                                               size_t partial_capacity,
                                               int threads);
void sq_avx2_rng_words(const sq_rng_stream *s, uint64_t n, uint32_t *out,
                       int threads);
sq_status sq_avx2_encode_payload(uint8_t bit_width, const float *values,
                                 size_t count, float scale,
                                 const uint32_t *words, uint8_t *payload,
                                 int threads);

#endif
