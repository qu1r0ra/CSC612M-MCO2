#ifndef SQ_QUANTIZER_H
#define SQ_QUANTIZER_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"

#define SQ_SCALE_BLOCK_SIZE 256

sq_status sq_validate_input(const float *values, size_t count);
sq_status sq_compute_scale(const float *values, size_t count, float *scale);
size_t sq_scale_workspace_elements(size_t count);
sq_status sq_compute_scale_with_workspace(const float *values, size_t count,
                                          float *scale, float *partials,
                                          size_t partial_capacity);
sq_status sq_encode_payload(uint8_t bit_width, const float *values,
                            size_t count, float scale, const uint32_t *words,
                            uint8_t *payload);
sq_status sq_q8_make_codes(const float *values, size_t count, float scale,
                           const uint32_t *words, uint8_t *codes);

#endif
