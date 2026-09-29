#ifndef SQ_CODEC_H
#define SQ_CODEC_H

#include <stddef.h>
#include <stdint.h>

#include "sq_status.h"

#define SQ_HEADER_SIZE 20
#define SQ_Q8_BITS 8
#define SQ_Q4_BITS 4
#define SQ_Q8_SIGNED_LIMIT 127
#define SQ_Q4_SIGNED_LIMIT 7

/* Payload bytes for a record of count codes; Q4 packs two codes per byte. */
static inline size_t sq_payload_size(uint8_t bit_width, size_t count) {
  return bit_width == SQ_Q4_BITS ? count / 2 + (count & 1) : count;
}

sq_status sq_header_encode(uint8_t bit_width, uint64_t count, float scale,
                           uint8_t header[SQ_HEADER_SIZE]);

/* Allocate decoded values on success; the caller releases them with free(). */
sq_status sq_decode_record(const uint8_t *record, size_t record_size,
                           float **values, size_t *count);

#endif
