#include "quantizer.h"

#include <math.h>
#include <stdlib.h>

#include "mco2_rng.h"

mco2_q8_status mco2_q8_validate_input(const float *values, size_t count)
{
    size_t i;
    if (count != 0 && values == NULL)
        return MCO2_Q8_ERR_ARGUMENT;
    for (i = 0; i < count; i++) {
        if (!isfinite(values[i]))
            return MCO2_Q8_ERR_NONFINITE;
    }
    return MCO2_Q8_OK;
}

static float reduce_block(const float *values, size_t count, size_t start,
                          float max_abs)
{
    float terms[MCO2_Q8_SCALE_BLOCK_SIZE];
    size_t remaining = count - start;
    size_t block_count = remaining < MCO2_Q8_SCALE_BLOCK_SIZE
                             ? remaining
                             : MCO2_Q8_SCALE_BLOCK_SIZE;
    size_t i, stride;

    for (i = 0; i < MCO2_Q8_SCALE_BLOCK_SIZE; i++)
        terms[i] = 0.0f;
    for (i = 0; i < block_count; i++) {
        float ratio = fabsf(values[start + i]) / max_abs;
        terms[i] = ratio * ratio;
    }

    for (stride = MCO2_Q8_SCALE_BLOCK_SIZE / 2; stride != 0; stride /= 2) {
        for (i = 0; i < stride; i++)
            terms[i] = terms[2 * i] + terms[2 * i + 1];
    }
    return terms[0];
}

mco2_q8_status mco2_q8_compute_scale(const float *values, size_t count,
                                     float *scale)
{
    float max_abs = 0.0f;
    float *partials;
    size_t block_count, padded_count = 1, start, stride, i;
    mco2_q8_status status;

    if (scale == NULL || (count != 0 && values == NULL))
        return MCO2_Q8_ERR_ARGUMENT;
    *scale = 0.0f;
    status = mco2_q8_validate_input(values, count);
    if (status != MCO2_Q8_OK)
        return status;
    if (count == 0)
        return MCO2_Q8_OK;

    for (i = 0; i < count; i++) {
        float magnitude = fabsf(values[i]);
        if (magnitude > max_abs)
            max_abs = magnitude;
    }
    if (max_abs == 0.0f)
        return MCO2_Q8_OK;

    block_count = count / MCO2_Q8_SCALE_BLOCK_SIZE;
    if (count % MCO2_Q8_SCALE_BLOCK_SIZE != 0)
        block_count++;
    while (padded_count < block_count) {
        if (padded_count > SIZE_MAX / 2)
            return MCO2_Q8_ERR_MEMORY;
        padded_count *= 2;
    }
    if (padded_count > SIZE_MAX / sizeof *partials)
        return MCO2_Q8_ERR_MEMORY;
    partials = (float *)calloc(padded_count, sizeof *partials);
    if (partials == NULL)
        return MCO2_Q8_ERR_MEMORY;

    for (i = 0, start = 0; i < block_count; i++, start += MCO2_Q8_SCALE_BLOCK_SIZE)
        partials[i] = reduce_block(values, count, start, max_abs);
    for (stride = padded_count / 2; stride != 0; stride /= 2) {
        for (i = 0; i < stride; i++)
            partials[i] = partials[2 * i] + partials[2 * i + 1];
    }

    {
        float root = sqrtf(partials[0]);
        float result = max_abs * root;
        free(partials);
        if (!isfinite(result))
            return MCO2_Q8_ERR_SCALE_OVERFLOW;
        *scale = result;
    }
    return MCO2_Q8_OK;
}

mco2_q8_status mco2_q8_make_codes(const float *values, size_t count,
                                   float scale, const uint32_t *words,
                                   uint8_t *codes)
{
    size_t i;
    int has_nonzero = 0;
    mco2_q8_status status;

    if (count != 0 && (values == NULL || words == NULL || codes == NULL))
        return MCO2_Q8_ERR_ARGUMENT;
    if (!isfinite(scale) || scale < 0.0f)
        return MCO2_Q8_ERR_SCALE;
    if (count == 0 && scale != 0.0f)
        return MCO2_Q8_ERR_SCALE;
    status = mco2_q8_validate_input(values, count);
    if (status != MCO2_Q8_OK)
        return status;
    for (i = 0; i < count; i++) {
        if (values[i] != 0.0f) {
            has_nonzero = 1;
            break;
        }
    }
    if (!has_nonzero && scale != 0.0f)
        return MCO2_Q8_ERR_SCALE;

    if (scale == 0.0f) {
        for (i = 0; i < count; i++) {
            if (values[i] != 0.0f)
                return MCO2_Q8_ERR_SCALE;
            codes[i] = MCO2_Q8_SIGNED_LIMIT;
        }
        return MCO2_Q8_OK;
    }

    for (i = 0; i < count; i++) {
        float absolute_value = fabsf(values[i]);
        float scaled = (absolute_value / scale) * (float)MCO2_Q8_SIGNED_LIMIT;
        float lower_float;
        float probability;
        int magnitude, signed_code;

        if (scaled > (float)MCO2_Q8_SIGNED_LIMIT)
            scaled = (float)MCO2_Q8_SIGNED_LIMIT;
        lower_float = floorf(scaled);
        probability = scaled - lower_float;
        magnitude = (int)lower_float + mco2_bernoulli(words[i], probability);
        signed_code = signbit(values[i]) && magnitude != 0 ? -magnitude : magnitude;
        codes[i] = (uint8_t)(signed_code + MCO2_Q8_SIGNED_LIMIT);
    }
    return MCO2_Q8_OK;
}
