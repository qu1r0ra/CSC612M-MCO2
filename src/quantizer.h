#ifndef MCO2_QUANTIZER_H
#define MCO2_QUANTIZER_H

#include <stddef.h>
#include <stdint.h>

#include "codec.h"

#define MCO2_Q8_SCALE_BLOCK_SIZE 256

mco2_q8_status mco2_q8_validate_input(const float *values, size_t count);
mco2_q8_status mco2_q8_compute_scale(const float *values, size_t count,
                                     float *scale);
mco2_q8_status mco2_q8_make_codes(const float *values, size_t count,
                                   float scale, const uint32_t *words,
                                   uint8_t *codes);

#endif
