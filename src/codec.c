#include "codec.h"
#include "byteorder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t mco2_q8_magic[4] = {'M', 'S', 'Q', '1'};

static void write_u64_le(uint8_t *out, uint64_t value)
{
    unsigned int i;
    for (i = 0; i < 8; i++)
        out[i] = (uint8_t)(value >> (i * 8));
}

static uint64_t read_u64_le(const uint8_t *in)
{
    uint64_t value = 0;
    unsigned int i;
    for (i = 0; i < 8; i++)
        value |= (uint64_t)in[i] << (i * 8);
    return value;
}

const char *mco2_q8_status_message(mco2_q8_status status)
{
    switch (status) {
    case MCO2_Q8_OK:
        return "ok";
    case MCO2_Q8_ERR_ARGUMENT:
        return "invalid argument";
    case MCO2_Q8_ERR_MAGIC:
        return "bad record magic";
    case MCO2_Q8_ERR_VERSION:
        return "unsupported record version";
    case MCO2_Q8_ERR_BIT_WIDTH:
        return "unsupported bit width; this tool accepts 8-bit records";
    case MCO2_Q8_ERR_RESERVED:
        return "reserved header bytes must be zero";
    case MCO2_Q8_ERR_COUNT:
        return "element count is too large";
    case MCO2_Q8_ERR_PAYLOAD_LENGTH:
        return "payload length does not match element count";
    case MCO2_Q8_ERR_SCALE:
        return "scale must be finite and non-negative; "
               "empty and all-zero inputs require zero scale";
    case MCO2_Q8_ERR_CODE:
        return "record contains an invalid 8-bit code";
    case MCO2_Q8_ERR_ZERO_SCALE_CODE:
        return "zero-scale records must contain only the center code";
    case MCO2_Q8_ERR_NONFINITE:
        return "input contains a non-finite FP32 value";
    case MCO2_Q8_ERR_SCALE_OVERFLOW:
        return "FP32 L2 scale overflow";
    case MCO2_Q8_ERR_ID_OVERFLOW:
        return "tensor and invocation identifiers must fit in uint32";
    case MCO2_Q8_ERR_MEMORY:
        return "unable to allocate memory";
    case MCO2_Q8_ERR_IO:
        return "file read or write failed";
    }
    return "unknown error";
}

mco2_q8_status mco2_q8_header_encode(uint8_t bit_width, uint64_t count,
                                     float scale,
                                     uint8_t header[MCO2_Q8_HEADER_SIZE])
{
    uint32_t scale_bits;

    if (header == NULL)
        return MCO2_Q8_ERR_ARGUMENT;
    if (bit_width != MCO2_Q8_BITS)
        return MCO2_Q8_ERR_BIT_WIDTH;
    if (!isfinite(scale) || scale < 0.0f)
        return MCO2_Q8_ERR_SCALE;
    if (count == 0 && scale != 0.0f)
        return MCO2_Q8_ERR_SCALE;

    memcpy(header, mco2_q8_magic, sizeof mco2_q8_magic);
    header[4] = 1;
    header[5] = bit_width;
    header[6] = 0;
    header[7] = 0;
    write_u64_le(header + 8, count);
    memcpy(&scale_bits, &scale, sizeof scale_bits);
    mco2_store_u32_le(header + 16, scale_bits);
    return MCO2_Q8_OK;
}

mco2_q8_status mco2_q8_decode_record(const uint8_t *record,
                                     size_t record_size, float **values,
                                     size_t *count)
{
    uint64_t count64;
    uint32_t scale_bits;
    float scale;
    size_t n, i;
    float *decoded = NULL;

    if (values == NULL || count == NULL || (record == NULL && record_size != 0))
        return MCO2_Q8_ERR_ARGUMENT;
    *values = NULL;
    *count = 0;
    if (record_size < MCO2_Q8_HEADER_SIZE)
        return MCO2_Q8_ERR_PAYLOAD_LENGTH;
    if (memcmp(record, mco2_q8_magic, sizeof mco2_q8_magic) != 0)
        return MCO2_Q8_ERR_MAGIC;
    if (record[4] != 1)
        return MCO2_Q8_ERR_VERSION;
    if (record[5] != MCO2_Q8_BITS)
        return MCO2_Q8_ERR_BIT_WIDTH;
    if (record[6] != 0 || record[7] != 0)
        return MCO2_Q8_ERR_RESERVED;

    count64 = read_u64_le(record + 8);
    if (count64 > (uint64_t)(SIZE_MAX - MCO2_Q8_HEADER_SIZE) ||
        count64 > (uint64_t)(SIZE_MAX / sizeof(float)))
        return MCO2_Q8_ERR_COUNT;
    n = (size_t)count64;
    if (record_size != MCO2_Q8_HEADER_SIZE + n)
        return MCO2_Q8_ERR_PAYLOAD_LENGTH;

    scale_bits = mco2_load_u32_le(record + 16);
    memcpy(&scale, &scale_bits, sizeof scale);
    if (!isfinite(scale) || scale < 0.0f || (n == 0 && scale != 0.0f))
        return MCO2_Q8_ERR_SCALE;

    if (n != 0) {
        decoded = (float *)malloc(n * sizeof *decoded);
        if (decoded == NULL)
            return MCO2_Q8_ERR_MEMORY;
    }
    for (i = 0; i < n; i++) {
        uint8_t code = record[MCO2_Q8_HEADER_SIZE + i];
        int signed_code;
        if (code > 2 * MCO2_Q8_SIGNED_LIMIT) {
            free(decoded);
            return MCO2_Q8_ERR_CODE;
        }
        if (scale == 0.0f && code != MCO2_Q8_SIGNED_LIMIT) {
            free(decoded);
            return MCO2_Q8_ERR_ZERO_SCALE_CODE;
        }
        signed_code = (int)code - MCO2_Q8_SIGNED_LIMIT;
        decoded[i] = ((float)signed_code / (float)MCO2_Q8_SIGNED_LIMIT) * scale;
    }

    *values = decoded;
    *count = n;
    return MCO2_Q8_OK;
}
