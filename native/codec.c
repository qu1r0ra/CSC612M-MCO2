#include "codec.h"
#include "byteorder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t sq_magic[4] = {'M', 'S', 'Q', '1'};

sq_status sq_header_encode(uint8_t bit_width, uint64_t count, float scale,
                           uint8_t header[SQ_HEADER_SIZE]) {
  uint32_t scale_bits;

  if (header == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  if (!isfinite(scale) || scale < 0.0f) {
    return SQ_ERR_SCALE;
  }
  if (count == 0 && scale != 0.0f) {
    return SQ_ERR_SCALE;
  }

  memcpy(header, sq_magic, sizeof sq_magic);
  header[4] = 1;
  header[5] = bit_width;
  header[6] = 0;
  header[7] = 0;
  sq_store_u64_le(header + 8, count);
  memcpy(&scale_bits, &scale, sizeof scale_bits);
  sq_store_u32_le(header + 16, scale_bits);
  return SQ_OK;
}

sq_status sq_decode_record(const uint8_t *record, size_t record_size,
                           float **values, size_t *count) {
  uint8_t bit_width;
  uint64_t count64;
  uint32_t scale_bits;
  float scale;
  size_t n, payload_len, i;
  float *decoded = NULL;

  if (values == NULL || count == NULL || (record == NULL && record_size != 0)) {
    return SQ_ERR_ARGUMENT;
  }
  *values = NULL;
  *count = 0;
  if (record_size < SQ_HEADER_SIZE) {
    return SQ_ERR_PAYLOAD_LENGTH;
  }
  if (memcmp(record, sq_magic, sizeof sq_magic) != 0) {
    return SQ_ERR_MAGIC;
  }
  if (record[4] != 1) {
    return SQ_ERR_VERSION;
  }
  bit_width = record[5];
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  if (record[6] != 0 || record[7] != 0) {
    return SQ_ERR_RESERVED;
  }

  count64 = sq_load_u64_le(record + 8);
  if (count64 > (uint64_t)(SIZE_MAX - SQ_HEADER_SIZE) ||
      count64 > (uint64_t)(SIZE_MAX / sizeof(float))) {
    return SQ_ERR_COUNT;
  }
  n = (size_t)count64;
  payload_len = (bit_width == SQ_Q4_BITS) ? (n + 1) / 2 : n;
  if (record_size != SQ_HEADER_SIZE + payload_len) {
    return SQ_ERR_PAYLOAD_LENGTH;
  }

  scale_bits = sq_load_u32_le(record + 16);
  memcpy(&scale, &scale_bits, sizeof scale);
  if (!isfinite(scale) || scale < 0.0f || (n == 0 && scale != 0.0f)) {
    return SQ_ERR_SCALE;
  }

  if (n != 0) {
    decoded = (float *)malloc(n * sizeof *decoded);
    if (decoded == NULL) {
      return SQ_ERR_MEMORY;
    }
  }
  if (bit_width == SQ_Q8_BITS) {
    for (i = 0; i < n; i++) {
      uint8_t code = record[SQ_HEADER_SIZE + i];
      int signed_code;
      if (code > 2 * SQ_Q8_SIGNED_LIMIT) {
        free(decoded);
        return SQ_ERR_CODE;
      }
      if (scale == 0.0f && code != SQ_Q8_SIGNED_LIMIT) {
        free(decoded);
        return SQ_ERR_ZERO_SCALE_CODE;
      }
      signed_code = (int)code - SQ_Q8_SIGNED_LIMIT;
      decoded[i] = ((float)signed_code / (float)SQ_Q8_SIGNED_LIMIT) * scale;
    }
  } else {
    const uint8_t *payload = record + SQ_HEADER_SIZE;
    if (n % 2 == 1) {
      uint8_t last_byte = payload[n / 2];
      if ((last_byte >> 4) != 0) {
        free(decoded);
        return SQ_ERR_PADDING_NIBBLE;
      }
    }
    for (i = 0; i < n; i++) {
      uint8_t byte_val = payload[i / 2];
      uint8_t code =
          (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
      int signed_code;
      if (code > 2 * SQ_Q4_SIGNED_LIMIT) {
        free(decoded);
        return SQ_ERR_CODE;
      }
      if (scale == 0.0f && code != SQ_Q4_SIGNED_LIMIT) {
        free(decoded);
        return SQ_ERR_ZERO_SCALE_CODE;
      }
      signed_code = (int)code - SQ_Q4_SIGNED_LIMIT;
      decoded[i] = ((float)signed_code / (float)SQ_Q4_SIGNED_LIMIT) * scale;
    }
  }

  *values = decoded;
  *count = n;
  return SQ_OK;
}
