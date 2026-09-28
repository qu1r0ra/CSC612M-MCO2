#ifndef SQ_CODEC_H
#define SQ_CODEC_H

#include <stddef.h>
#include <stdint.h>

#define SQ_HEADER_SIZE 20
#define SQ_Q8_BITS 8
#define SQ_Q4_BITS 4
#define SQ_Q8_SIGNED_LIMIT 127
#define SQ_Q4_SIGNED_LIMIT 7

typedef enum {
    SQ_OK = 0,
    SQ_ERR_ARGUMENT,
    SQ_ERR_MAGIC,
    SQ_ERR_VERSION,
    SQ_ERR_BIT_WIDTH,
    SQ_ERR_RESERVED,
    SQ_ERR_COUNT,
    SQ_ERR_PAYLOAD_LENGTH,
    SQ_ERR_SCALE,
    SQ_ERR_CODE,
    SQ_ERR_ZERO_SCALE_CODE,
    SQ_ERR_PADDING_NIBBLE,
    SQ_ERR_NONFINITE,
    SQ_ERR_SCALE_OVERFLOW,
    SQ_ERR_ID_OVERFLOW,
    SQ_ERR_MEMORY,
    SQ_ERR_IO,
    SQ_ERR_CUDA_UNAVAILABLE,
    SQ_ERR_CUDA,
    SQ_ERR_CUDA_BIT_WIDTH,
    SQ_ERR_TIMINGS_BACKEND,
    SQ_ERR_CLOCK,
    SQ_ERR_AVX2_UNAVAILABLE
} sq_status;

const char *sq_status_message(sq_status status);

sq_status sq_header_encode(uint8_t bit_width, uint64_t count,
                           float scale,
                           uint8_t header[SQ_HEADER_SIZE]);

/* Allocate decoded values on success; the caller releases them with free(). */
sq_status sq_decode_record(const uint8_t *record,
                           size_t record_size, float **values,
                           size_t *count);

#endif
