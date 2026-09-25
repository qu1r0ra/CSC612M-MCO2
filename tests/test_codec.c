#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec.h"

static int failures;

static void check(int condition, const char *description)
{
    printf("%s %s\n", condition ? "ok  " : "FAIL", description);
    if (!condition)
        failures++;
}

int main(void)
{
    static const uint8_t expected[MCO2_Q8_HEADER_SIZE] = {
        0x4d, 0x53, 0x51, 0x31, 0x01, 0x08, 0x00, 0x00,
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        0x00, 0x00, 0x80, 0x3f,
    };
    uint8_t header[MCO2_Q8_HEADER_SIZE];

    check(mco2_q8_header_encode(8, UINT64_C(0x0102030405060708), 1.0f,
                                header) == MCO2_Q8_OK,
          "20-byte header serializes");
    check(memcmp(header + 0, expected + 0, 4) == 0, "magic at offset 0, size 4");
    check(header[4] == expected[4], "version at offset 4, size 1");
    check(header[5] == expected[5], "bit width at offset 5, size 1");
    check(memcmp(header + 6, expected + 6, 2) == 0,
          "reserved bytes at offset 6, size 2");
    check(memcmp(header + 8, expected + 8, 8) == 0,
          "element count at offset 8, size 8, little-endian");
    check(memcmp(header + 16, expected + 16, 4) == 0,
          "FP32 scale at offset 16, size 4, little-endian");
    check(memcmp(header, expected, sizeof expected) == 0,
          "complete header matches the frozen 20-byte layout");
    check(mco2_q8_header_encode(8, 0, 1.0f, header) == MCO2_Q8_ERR_SCALE,
          "empty records require zero scale");

    /* 4-bit header serialization */
    check(mco2_q8_header_encode(4, 5, 2.0f, header) == MCO2_Q8_OK &&
              header[5] == 4,
          "4-bit header encodes bit width 4");

    /* Decoder rejection tests: one test per check */
    {
        /* 1. Out-of-range 8-bit code (> 254) */
        uint8_t rec_q8_bad_code[MCO2_Q8_HEADER_SIZE + 1];
        float *vals = NULL;
        size_t count = 0;
        mco2_q8_header_encode(8, 1, 1.0f, rec_q8_bad_code);
        rec_q8_bad_code[MCO2_Q8_HEADER_SIZE] = 255;
        check(mco2_q8_decode_record(rec_q8_bad_code, sizeof rec_q8_bad_code,
                                    &vals, &count) == MCO2_Q8_ERR_CODE,
              "decoder rejects out-of-range 8-bit code");

        /* 2. Out-of-range 4-bit code (> 14) */
        uint8_t rec_q4_bad_code[MCO2_Q8_HEADER_SIZE + 1];
        mco2_q8_header_encode(4, 1, 1.0f, rec_q4_bad_code);
        rec_q4_bad_code[MCO2_Q8_HEADER_SIZE] = 0x0F; /* low nibble is 15 */
        check(mco2_q8_decode_record(rec_q4_bad_code, sizeof rec_q4_bad_code,
                                    &vals, &count) == MCO2_Q8_ERR_CODE,
              "decoder rejects out-of-range 4-bit code");

        /* 3. Nonzero padding nibble in odd-length 4-bit payload */
        uint8_t rec_q4_bad_pad[MCO2_Q8_HEADER_SIZE + 1];
        mco2_q8_header_encode(4, 1, 1.0f, rec_q4_bad_pad);
        rec_q4_bad_pad[MCO2_Q8_HEADER_SIZE] = 0x17; /* low nibble 7, high nibble 1 */
        check(mco2_q8_decode_record(rec_q4_bad_pad, sizeof rec_q4_bad_pad,
                                    &vals, &count) == MCO2_Q8_ERR_PADDING_NIBBLE,
              "decoder rejects nonzero padding nibble");

        /* 4. Nonfinite scale */
        uint8_t rec_nan_scale[MCO2_Q8_HEADER_SIZE + 1];
        uint32_t nan_bits = 0x7FC00000;
        mco2_q8_header_encode(8, 1, 1.0f, rec_nan_scale);
        memcpy(rec_nan_scale + 16, &nan_bits, sizeof nan_bits);
        rec_nan_scale[MCO2_Q8_HEADER_SIZE] = 127;
        check(mco2_q8_decode_record(rec_nan_scale, sizeof rec_nan_scale,
                                    &vals, &count) == MCO2_Q8_ERR_SCALE,
              "decoder rejects nonfinite scale");

        /* 5. Negative scale */
        uint8_t rec_neg_scale[MCO2_Q8_HEADER_SIZE + 1];
        float neg_scale = -1.0f;
        mco2_q8_header_encode(8, 1, 1.0f, rec_neg_scale);
        memcpy(rec_neg_scale + 16, &neg_scale, sizeof neg_scale);
        rec_neg_scale[MCO2_Q8_HEADER_SIZE] = 127;
        check(mco2_q8_decode_record(rec_neg_scale, sizeof rec_neg_scale,
                                    &vals, &count) == MCO2_Q8_ERR_SCALE,
              "decoder rejects negative scale");

        /* 6. Zero-scale 8-bit record with code other than s */
        uint8_t rec_q8_zero_scale_bad[MCO2_Q8_HEADER_SIZE + 1];
        mco2_q8_header_encode(8, 1, 0.0f, rec_q8_zero_scale_bad);
        rec_q8_zero_scale_bad[MCO2_Q8_HEADER_SIZE] = 0; /* s is 127 */
        check(mco2_q8_decode_record(rec_q8_zero_scale_bad,
                                    sizeof rec_q8_zero_scale_bad, &vals,
                                    &count) == MCO2_Q8_ERR_ZERO_SCALE_CODE,
              "decoder rejects zero-scale 8-bit record with code other than s");

        /* 7. Zero-scale 4-bit record with code other than s */
        uint8_t rec_q4_zero_scale_bad[MCO2_Q8_HEADER_SIZE + 1];
        mco2_q8_header_encode(4, 1, 0.0f, rec_q4_zero_scale_bad);
        rec_q4_zero_scale_bad[MCO2_Q8_HEADER_SIZE] = 0x00; /* s is 7 */
        check(mco2_q8_decode_record(rec_q4_zero_scale_bad,
                                    sizeof rec_q4_zero_scale_bad, &vals,
                                    &count) == MCO2_Q8_ERR_ZERO_SCALE_CODE,
              "decoder rejects zero-scale 4-bit record with code other than s");

        /* 8. Valid 4-bit round trip decode */
        uint8_t rec_q4_valid[MCO2_Q8_HEADER_SIZE + 2];
        mco2_q8_header_encode(4, 3, 2.0f, rec_q4_valid);
        /* elements: code 0 (k=-7 -> -2.0), code 7 (k=0 -> 0.0), code 14 (k=7 -> 2.0) */
        rec_q4_valid[MCO2_Q8_HEADER_SIZE + 0] = (uint8_t)(0x00 | (0x07 << 4));
        rec_q4_valid[MCO2_Q8_HEADER_SIZE + 1] = (uint8_t)(0x0E | (0x00 << 4)); /* pad 0 */
        check(mco2_q8_decode_record(rec_q4_valid, sizeof rec_q4_valid, &vals,
                                    &count) == MCO2_Q8_OK &&
                  count == 3 && vals[0] == -2.0f && vals[1] == 0.0f &&
                  vals[2] == 2.0f,
              "valid 4-bit record decodes with correct elements and zero padding");
        free(vals);
    }

    printf("# %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
