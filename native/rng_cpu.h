#ifndef SQ_RNG_CPU_H
#define SQ_RNG_CPU_H

#include "sq_rng.h"

/* out[i] = word for element i under the logical mapping. */
void sq_rng_words_cpu(const sq_rng_stream *s, uint64_t n, uint32_t *out);

#endif
