#ifndef MCO2_CODEC_H
#define MCO2_CODEC_H

#include <stddef.h>
#include <stdint.h>

#define MCO2_Q8_HEADER_SIZE 20
#define MCO2_Q8_BITS 8
#define MCO2_Q4_BITS 4
#define MCO2_Q8_SIGNED_LIMIT 127
#define MCO2_Q4_SIGNED_LIMIT 7

typedef enum {
    MCO2_Q8_OK = 0,
    MCO2_Q8_ERR_ARGUMENT,
    MCO2_Q8_ERR_MAGIC,
    MCO2_Q8_ERR_VERSION,
    MCO2_Q8_ERR_BIT_WIDTH,
    MCO2_Q8_ERR_RESERVED,
    MCO2_Q8_ERR_COUNT,
    MCO2_Q8_ERR_PAYLOAD_LENGTH,
    MCO2_Q8_ERR_SCALE,
    MCO2_Q8_ERR_CODE,
    MCO2_Q8_ERR_ZERO_SCALE_CODE,
    MCO2_Q8_ERR_PADDING_NIBBLE,
    MCO2_Q8_ERR_NONFINITE,
    MCO2_Q8_ERR_SCALE_OVERFLOW,
    MCO2_Q8_ERR_ID_OVERFLOW,
    MCO2_Q8_ERR_MEMORY,
    MCO2_Q8_ERR_IO,
    MCO2_Q8_ERR_CUDA_UNAVAILABLE,
    MCO2_Q8_ERR_CUDA,
    MCO2_Q8_ERR_CUDA_BIT_WIDTH,
    MCO2_Q8_ERR_TIMINGS_BACKEND
} mco2_q8_status;

const char *mco2_q8_status_message(mco2_q8_status status);

mco2_q8_status mco2_q8_header_encode(uint8_t bit_width, uint64_t count,
                                     float scale,
                                     uint8_t header[MCO2_Q8_HEADER_SIZE]);

/* Allocate decoded values on success; the caller releases them with free(). */
mco2_q8_status mco2_q8_decode_record(const uint8_t *record,
                                     size_t record_size, float **values,
                                     size_t *count);

#endif
