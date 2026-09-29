#ifndef SQ_BYTEORDER_H
#define SQ_BYTEORDER_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint32_t sq_load_u32_le(const uint8_t bytes[4]) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static inline void sq_store_u32_le(uint8_t bytes[4], uint32_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
  bytes[2] = (uint8_t)(value >> 16);
  bytes[3] = (uint8_t)(value >> 24);
}

static inline void sq_store_u64_le(uint8_t bytes[8], uint64_t value) {
  sq_store_u32_le(bytes, (uint32_t)value);
  sq_store_u32_le(bytes + 4, (uint32_t)(value >> 32));
}

static inline uint64_t sq_load_u64_le(const uint8_t bytes[8]) {
  return (uint64_t)sq_load_u32_le(bytes) |
         ((uint64_t)sq_load_u32_le(bytes + 4) << 32);
}

static inline void sq_load_u32_array_le(uint32_t *out, const uint8_t *bytes,
                                        size_t count) {
  size_t i;
  for (i = 0; i < count; i++) {
    out[i] = sq_load_u32_le(bytes + 4 * i);
  }
}

static inline void sq_load_f32_array_le(float *out, const uint8_t *bytes,
                                        size_t count) {
  size_t i;
  for (i = 0; i < count; i++) {
    uint32_t bits = sq_load_u32_le(bytes + 4 * i);
    memcpy(&out[i], &bits, sizeof bits);
  }
}

static inline void sq_store_f32_array_le(uint8_t *bytes, const float *values,
                                         size_t count) {
  size_t i;
  for (i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, &values[i], sizeof bits);
    sq_store_u32_le(bytes + 4 * i, bits);
  }
}

#endif
