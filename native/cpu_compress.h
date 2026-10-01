#ifndef SQ_CPU_COMPRESS_H
#define SQ_CPU_COMPRESS_H

#include <stddef.h>
#include <stdint.h>

#include "bench_types.h"
#include "sq_status.h"

typedef struct {
  uint32_t *generated_words;
  size_t generated_word_capacity;
  float *scale_partials;
  size_t scale_partial_capacity;
} sq_cpu_compress_workspace;

void sq_cpu_compress_workspace_init(sq_cpu_compress_workspace *workspace);
/* Reserve and validate once before entering repeated compression calls. */
sq_status
sq_cpu_compress_workspace_reserve(sq_cpu_compress_workspace *workspace,
                                  sq_backend backend, int threads,
                                  size_t count);
void sq_cpu_compress_workspace_destroy(sq_cpu_compress_workspace *workspace);

/* Requires a workspace reserved for this backend, thread count, and capacity.
 */
sq_status sq_cpu_compress(sq_backend backend, uint8_t bit_width,
                          const float *values, size_t count, uint64_t seed,
                          uint64_t tensor_id, uint64_t invocation_id,
                          int prescribed_scale_seen, float prescribed_scale,
                          const uint32_t *prescribed_words, int threads,
                          sq_cpu_compress_workspace *workspace,
                          uint8_t *record);

#endif
