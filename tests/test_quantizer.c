#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "quantizer.h"

static int failures;

static void check(int condition, const char *description)
{
    printf("%s %s\n", condition ? "ok  " : "FAIL", description);
    if (!condition)
        failures++;
}

int main(void)
{
    static const float vector[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    static const uint32_t words[] = {0, 0, 0, UINT32_MAX, 0};
    static const uint8_t expected[] = {0x00, 0x3f, 0x7f, 0xbe, 0xfe};
    float scale;
    uint8_t codes[sizeof vector / sizeof vector[0]];
    const float three_four[] = {3.0f, 4.0f};
    const float zeros[] = {0.0f, -0.0f, 0.0f};
    const float nonfinite[] = {1.0f, NAN};
    const float overflow[] = {FLT_MAX, FLT_MAX};

    check(mco2_q8_compute_scale(NULL, 0, &scale) == MCO2_Q8_OK && scale == 0.0f,
          "empty input has zero scale");
    check(mco2_q8_compute_scale(zeros, sizeof zeros / sizeof zeros[0], &scale) ==
                  MCO2_Q8_OK &&
              scale == 0.0f,
          "all-zero input has zero scale");
    check(mco2_q8_compute_scale(three_four, 2, &scale) == MCO2_Q8_OK &&
              scale == 5.0f,
          "max-rescaled fixed-order scale for [3, 4]");
    check(mco2_q8_compute_scale(nonfinite, 2, &scale) == MCO2_Q8_ERR_NONFINITE,
          "non-finite input is rejected");
    check(mco2_q8_compute_scale(overflow, 2, &scale) ==
              MCO2_Q8_ERR_SCALE_OVERFLOW,
          "unrepresentable FP32 norm is rejected");

    check(mco2_q8_make_codes(vector, 5, 2.0f, words, codes) == MCO2_Q8_OK &&
              memcmp(codes, expected, sizeof expected) == 0,
          "prescribed scale and words produce signed 8-bit codes");
    check(mco2_q8_make_codes(zeros, sizeof zeros / sizeof zeros[0], 0.0f,
                             (const uint32_t[]){0, 1, 2}, codes) == MCO2_Q8_OK &&
              codes[0] == 127 && codes[1] == 127 && codes[2] == 127,
          "zero-scale input maps to the center code");
    check(mco2_q8_make_codes(zeros, sizeof zeros / sizeof zeros[0], 1.0f,
                             (const uint32_t[]){0, 1, 2}, codes) ==
              MCO2_Q8_ERR_SCALE,
          "all-zero inputs reject a nonzero scale");

    printf("# %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
