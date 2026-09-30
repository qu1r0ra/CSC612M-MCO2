#ifndef SQ_QUANTIZER_H
#define SQ_QUANTIZER_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"

#define SQ_SCALE_BLOCK_SIZE 256

typedef sq_status (*sq_encode_input_scan)(const float *values, size_t count,
                                          void *context, int *has_nonzero);

typedef struct {
  int signed_limit;
  int skip_encoding;
} sq_encode_payload_state;

sq_status sq_validate_input(const float *values, size_t count);
sq_status sq_compute_scale(const float *values, size_t count, float *scale);
size_t sq_scale_workspace_elements(size_t count);
sq_status sq_compute_scale_with_workspace(const float *values, size_t count,
                                          float *scale, float *partials,
                                          size_t partial_capacity);
sq_status sq_encode_payload_prologue(uint8_t bit_width, const float *values,
                                     size_t count, float scale,
                                     const uint32_t *words, uint8_t *payload,
                                     sq_encode_input_scan scan,
                                     void *scan_context,
                                     sq_encode_payload_state *state);
sq_status sq_encode_payload(uint8_t bit_width, const float *values,
                            size_t count, float scale, const uint32_t *words,
                            uint8_t *payload);

#endif
