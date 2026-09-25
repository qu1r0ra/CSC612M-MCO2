#include <stdint.h>
#include <stdio.h>
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

    printf("# %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
