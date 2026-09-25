#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quantizer.h"
#include "rng_cpu.h"

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

    /* 4-bit prescribed scale and word output with odd payload length (5 elements -> 3 bytes) */
    {
        uint8_t q4_payload[3];
        static const uint8_t q4_expected[] = {0x30, 0xa7, 0x0e};
        check(mco2_encode_payload(4, vector, 5, 2.0f, words, q4_payload) ==
                      MCO2_Q8_OK &&
                  memcmp(q4_payload, q4_expected, sizeof q4_expected) == 0,
              "4-bit output puts lower index in low nibble and zeroes high nibble");
    }

    /* Edge case: signed zero maps to center code and decodes to +0 */
    {
        float sz[2] = {-0.0f, 1.0f};
        uint8_t q8_sz_payload[2];
        uint8_t q4_sz_payload[1];
        uint8_t record[MCO2_Q8_HEADER_SIZE + 2];
        float *decoded = NULL;
        size_t count = 0;

        check(mco2_encode_payload(8, sz, 2, 1.0f, words, q8_sz_payload) ==
                      MCO2_Q8_OK &&
                  q8_sz_payload[0] == 127 && q8_sz_payload[1] == 254,
              "8-bit signed zero encodes to center code 127");
        mco2_q8_header_encode(8, 2, 1.0f, record);
        memcpy(record + MCO2_Q8_HEADER_SIZE, q8_sz_payload, 2);
        check(mco2_q8_decode_record(record, sizeof record, &decoded, &count) ==
                      MCO2_Q8_OK &&
                  !signbit(decoded[0]) && decoded[0] == 0.0f &&
                  decoded[1] == 1.0f,
              "8-bit signed zero decodes to +0.0f");
        free(decoded);

        check(mco2_encode_payload(4, sz, 2, 1.0f, words, q4_sz_payload) ==
                      MCO2_Q8_OK &&
                  (q4_sz_payload[0] & 0x0F) == 7 &&
                  ((q4_sz_payload[0] >> 4) & 0x0F) == 14,
              "4-bit signed zero encodes to center code 7 in low nibble");
        mco2_q8_header_encode(4, 2, 1.0f, record);
        memcpy(record + MCO2_Q8_HEADER_SIZE, q4_sz_payload, 1);
        check(mco2_q8_decode_record(record, MCO2_Q8_HEADER_SIZE + 1, &decoded,
                                    &count) == MCO2_Q8_OK &&
                  !signbit(decoded[0]) && decoded[0] == 0.0f &&
                  decoded[1] == 1.0f,
              "4-bit signed zero decodes to +0.0f");
        free(decoded);
    }

    /* Edge case: largest representable magnitudes without intermediate overflow */
    {
        float large_vals[3] = {1.0e20f, -1.0e20f, 1.0e20f};
        float large_scale = 0.0f;
        uint8_t large_payload[3];
        check(mco2_q8_compute_scale(large_vals, 3, &large_scale) == MCO2_Q8_OK &&
                  isfinite(large_scale) && large_scale > 1.0e20f,
              "max-rescaled scale computes without intermediate overflow");
        check(mco2_encode_payload(8, large_vals, 3, large_scale, words,
                                  large_payload) == MCO2_Q8_OK,
              "large magnitudes encode without intermediate overflow");
    }

    /* Edge case: saturation boundaries clamp to 2*s */
    {
        float sat_vals[2] = {100.0f, -100.0f};
        uint8_t q8_sat[2];
        uint8_t q4_sat[1];
        check(mco2_encode_payload(8, sat_vals, 2, 1.0f, words, q8_sat) ==
                      MCO2_Q8_OK &&
                  q8_sat[0] == 254 && q8_sat[1] == 0,
              "8-bit saturation clamps to 0 and 254 (2*s)");
        check(mco2_encode_payload(4, sat_vals, 2, 1.0f, words, q4_sat) ==
                      MCO2_Q8_OK &&
                  (q4_sat[0] & 0x0F) == 14 && ((q4_sat[0] >> 4) & 0x0F) == 0,
              "4-bit saturation clamps to 0 and 14 (2*s)");
    }

    /* Edge case: non-multiple length (5, 257) round trips */
    {
        static const size_t odd_lengths[] = {1, 3, 5, 257};
        size_t l_idx;
        for (l_idx = 0; l_idx < sizeof odd_lengths / sizeof odd_lengths[0];
             l_idx++) {
            size_t n = odd_lengths[l_idx];
            float *test_in = (float *)malloc(n * sizeof *test_in);
            uint32_t *test_words = (uint32_t *)calloc(n, sizeof *test_words);
            uint8_t *rec = (uint8_t *)malloc(MCO2_Q8_HEADER_SIZE + (n + 1) / 2);
            float *out_vals = NULL;
            size_t out_count = 0;
            size_t k;

            for (k = 0; k < n; k++)
                test_in[k] = (float)(k + 1);
            check(mco2_q8_header_encode(4, (uint64_t)n, (float)(n + 1), rec) ==
                      MCO2_Q8_OK,
                  "non-multiple length header encode");
            check(mco2_encode_payload(4, test_in, n, (float)(n + 1), test_words,
                                      rec + MCO2_Q8_HEADER_SIZE) == MCO2_Q8_OK,
                  "non-multiple length payload encode");
            check(mco2_q8_decode_record(rec, MCO2_Q8_HEADER_SIZE + (n + 1) / 2,
                                        &out_vals,
                                        &out_count) == MCO2_Q8_OK &&
                      out_count == n,
                  "non-multiple length decodes without error");
            free(out_vals);
            free(rec);
            free(test_words);
            free(test_in);
        }
    }

    /* Layer 3 (course subset): fixed 1024-element input over 4096 fixed seeds */
    {
        const size_t l3_n = 1024;
        const size_t l3_t = 4096;
        float *x = (float *)malloc(l3_n * sizeof *x);
        double *sum_decoded = (double *)calloc(l3_n, sizeof *sum_decoded);
        uint32_t *l3_words = (uint32_t *)malloc(l3_n * sizeof *l3_words);
        uint8_t *l3_payload = (uint8_t *)malloc(l3_n);
        size_t idx, k, seed;
        int bits_case;

        /* Construct fixed 1024-element vector */
        x[0] = 0.0f;
        x[1] = -0.0f;
        x[2] = 2.0f;  /* saturation boundary */
        x[3] = -2.0f; /* negative saturation boundary */
        idx = 4;
        /* 4-bit exact points */
        for (k = 1; k < 7; k++) {
            x[idx++] = 2.0f * ((float)k / 7.0f);
            x[idx++] = -2.0f * ((float)k / 7.0f);
        }
        /* 8-bit exact points */
        for (k = 1; k < 127; k++) {
            x[idx++] = 2.0f * ((float)k / 127.0f);
            x[idx++] = -2.0f * ((float)k / 127.0f);
        }
        /* Varied fractional parts */
        for (k = idx; k < l3_n; k++) {
            x[k] = -1.95f + 3.9f * ((float)(k - idx) / (float)(l3_n - idx - 1));
        }

        for (bits_case = 0; bits_case < 2; bits_case++) {
            uint8_t b = bits_case == 0 ? 4 : 8;
            int s = bits_case == 0 ? 7 : 127;
            int all_bounds_passed = 1;
            int all_exact_passed = 1;

            memset(sum_decoded, 0, l3_n * sizeof *sum_decoded);
            for (seed = 0; seed < l3_t; seed++) {
                mco2_rng_stream stream;
                mco2_rng_stream_init(&stream, (uint64_t)seed, 0, 0);
                mco2_rng_words_cpu(&stream, (uint64_t)l3_n, l3_words);
                mco2_encode_payload(b, x, l3_n, 2.0f, l3_words, l3_payload);

                if (b == 8) {
                    for (k = 0; k < l3_n; k++) {
                        int signed_code = (int)l3_payload[k] - 127;
                        float dec = ((float)signed_code / 127.0f) * 2.0f;
                        sum_decoded[k] += (double)dec;
                    }
                } else {
                    for (k = 0; k < l3_n; k++) {
                        uint8_t byte_val = l3_payload[k / 2];
                        uint8_t nibble = (k % 2 == 0)
                                             ? (byte_val & 0x0F)
                                             : ((byte_val >> 4) & 0x0F);
                        int signed_code = (int)nibble - 7;
                        float dec = ((float)signed_code / 7.0f) * 2.0f;
                        sum_decoded[k] += (double)dec;
                    }
                }
            }

            for (k = 0; k < l3_n; k++) {
                double mean = sum_decoded[k] / (double)l3_t;
                double diff = fabs(mean - (double)x[k]);
                float scaled = (fabsf(x[k]) / 2.0f) * (float)s;
                float lower_val;
                float p;
                double bound;

                if (scaled > (float)s)
                    scaled = (float)s;
                lower_val = floorf(scaled);
                p = scaled - lower_val;

                if (p == 0.0f || p == 1.0f) {
                    float mean_f = (float)mean;
                    if (mean_f != x[k])
                        all_exact_passed = 0;
                } else {
                    bound = 5.0 * (2.0 / (double)s) *
                            sqrt((double)p * (1.0 - (double)p) / (double)l3_t);
                    if (diff > bound + 1e-6)
                        all_bounds_passed = 0;
                }
            }

            check(all_exact_passed,
                  b == 4 ? "Layer 3 4-bit exact equality where p is 0 or 1"
                         : "Layer 3 8-bit exact equality where p is 0 or 1");
            check(all_bounds_passed,
                  b == 4 ? "Layer 3 4-bit 5-sigma unbiasedness over 4096 seeds"
                         : "Layer 3 8-bit 5-sigma unbiasedness over 4096 seeds");
        }

        free(l3_payload);
        free(l3_words);
        free(sum_decoded);
        free(x);
    }

    printf("# %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
