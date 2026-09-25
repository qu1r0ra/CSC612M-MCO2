#ifndef MCO2_QUANTIZER_H
#define MCO2_QUANTIZER_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"

#define MCO2_Q8_SCALE_BLOCK_SIZE 256

mco2_q8_status mco2_q8_validate_input(const float *values, size_t count);
mco2_q8_status mco2_q8_compute_scale(const float *values, size_t count,
                                     float *scale);
size_t mco2_q8_scale_workspace_elements(size_t count);
mco2_q8_status mco2_q8_compute_scale_with_workspace(
    const float *values, size_t count, float *scale, float *partials,
    size_t partial_capacity);
mco2_q8_status mco2_encode_payload(uint8_t bit_width, const float *values,
                                   size_t count, float scale,
                                   const uint32_t *words, uint8_t *payload);
mco2_q8_status mco2_q8_make_codes(const float *values, size_t count,
                                   float scale, const uint32_t *words,
                                   uint8_t *codes);

#endif
