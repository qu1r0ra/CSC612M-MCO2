#include "quantizer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sq_rng.h"

sq_status sq_validate_input(const float *values, size_t count) {
  size_t i;
  if (count != 0 && values == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  for (i = 0; i < count; i++) {
    if (!isfinite(values[i])) {
      return SQ_ERR_NONFINITE;
    }
  }
  return SQ_OK;
}

static float reduce_block(const float *values, size_t count, size_t start,
                          float max_abs) {
  float terms[SQ_SCALE_BLOCK_SIZE];
  size_t remaining = count - start;
  size_t block_count =
      remaining < SQ_SCALE_BLOCK_SIZE ? remaining : SQ_SCALE_BLOCK_SIZE;
  size_t i, stride;

  for (i = 0; i < SQ_SCALE_BLOCK_SIZE; i++) {
    terms[i] = 0.0f;
  }
  for (i = 0; i < block_count; i++) {
    float ratio = fabsf(values[start + i]) / max_abs;
    terms[i] = ratio * ratio;
  }

  for (stride = SQ_SCALE_BLOCK_SIZE / 2; stride != 0; stride /= 2) {
    for (i = 0; i < stride; i++) {
      terms[i] = terms[2 * i] + terms[2 * i + 1];
    }
  }
  return terms[0];
}

size_t sq_scale_workspace_elements(size_t count) {
  size_t block_count, padded_count = 1;

  if (count == 0) {
    return 0;
  }
  block_count = count / SQ_SCALE_BLOCK_SIZE;
  if (count % SQ_SCALE_BLOCK_SIZE != 0) {
    block_count++;
  }
  while (padded_count < block_count) {
    if (padded_count > SIZE_MAX / 2) {
      return SIZE_MAX;
    }
    padded_count *= 2;
  }
  return padded_count;
}

sq_status sq_compute_scale_with_workspace(const float *values, size_t count,
                                          float *scale, float *partials,
                                          size_t partial_capacity) {
  float max_abs = 0.0f;
  size_t block_count, padded_count, start, stride, i;

  if (scale == NULL || (count != 0 && values == NULL)) {
    return SQ_ERR_ARGUMENT;
  }
  *scale = 0.0f;
  if (count == 0) {
    return SQ_OK;
  }

  for (i = 0; i < count; i++) {
    float magnitude;

    if (!isfinite(values[i])) {
      return SQ_ERR_NONFINITE;
    }
    magnitude = fabsf(values[i]);
    if (magnitude > max_abs) {
      max_abs = magnitude;
    }
  }
  if (max_abs == 0.0f) {
    return SQ_OK;
  }

  padded_count = sq_scale_workspace_elements(count);
  if (padded_count == SIZE_MAX) {
    return SQ_ERR_MEMORY;
  }
  block_count = count / SQ_SCALE_BLOCK_SIZE;
  if (count % SQ_SCALE_BLOCK_SIZE != 0) {
    block_count++;
  }
  if (padded_count > SIZE_MAX / sizeof *partials) {
    return SQ_ERR_MEMORY;
  }
  if (partials == NULL || partial_capacity < padded_count) {
    return SQ_ERR_ARGUMENT;
  }
  memset(partials, 0, padded_count * sizeof *partials);

  for (i = 0, start = 0; i < block_count; i++, start += SQ_SCALE_BLOCK_SIZE) {
    partials[i] = reduce_block(values, count, start, max_abs);
  }
  for (stride = padded_count / 2; stride != 0; stride /= 2) {
    for (i = 0; i < stride; i++) {
      // 2 * i + 1 < 2 * stride <= padded_count; the analyzer cannot divide.
      // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
      partials[i] = partials[2 * i] + partials[2 * i + 1];
    }
  }

  {
    float root = sqrtf(partials[0]);
    float result = max_abs * root;
    if (!isfinite(result)) {
      return SQ_ERR_SCALE_OVERFLOW;
    }
    *scale = result;
  }
  return SQ_OK;
}

sq_status sq_compute_scale(const float *values, size_t count, float *scale) {
  size_t workspace_elements;
  float *partials = NULL;
  sq_status status;

  if (scale == NULL || (count != 0 && values == NULL)) {
    return SQ_ERR_ARGUMENT;
  }
  workspace_elements = sq_scale_workspace_elements(count);
  if (workspace_elements == SIZE_MAX ||
      workspace_elements > SIZE_MAX / sizeof *partials) {
    return SQ_ERR_MEMORY;
  }
  if (workspace_elements != 0) {
    partials = (float *)malloc(workspace_elements * sizeof *partials);
    if (partials == NULL) {
      return SQ_ERR_MEMORY;
    }
  }
  status = sq_compute_scale_with_workspace(values, count, scale, partials,
                                           workspace_elements);
  free(partials);
  return status;
}

static sq_status scan_scalar_encode_input(const float *values, size_t count,
                                          void *context, int *has_nonzero) {
  size_t i;
  sq_status status;
  (void)context;

  status = sq_validate_input(values, count);
  if (status != SQ_OK) {
    return status;
  }
  *has_nonzero = 0;
  for (i = 0; i < count; i++) {
    if (values[i] != 0.0f) {
      *has_nonzero = 1;
      break;
    }
  }
  return SQ_OK;
}

sq_status sq_encode_payload_prologue(uint8_t bit_width, const float *values,
                                     size_t count, float scale,
                                     const uint32_t *words, uint8_t *payload,
                                     sq_encode_input_scan scan,
                                     void *scan_context,
                                     sq_encode_payload_state *state) {
  size_t i;
  int has_nonzero = 0;
  sq_status status;

  if (state == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  state->skip_encoding = 0;
  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  state->signed_limit =
      (bit_width == SQ_Q4_BITS) ? SQ_Q4_SIGNED_LIMIT : SQ_Q8_SIGNED_LIMIT;

  if (count != 0 && (values == NULL || words == NULL || payload == NULL)) {
    return SQ_ERR_ARGUMENT;
  }
  if (!isfinite(scale) || scale < 0.0f) {
    return SQ_ERR_SCALE;
  }
  if (count == 0 && scale != 0.0f) {
    return SQ_ERR_SCALE;
  }
  if (count == 0) {
    state->skip_encoding = 1;
    return SQ_OK;
  }
  if (scan == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  status = scan(values, count, scan_context, &has_nonzero);
  if (status != SQ_OK) {
    return status;
  }
  if (!has_nonzero && scale != 0.0f) {
    return SQ_ERR_SCALE;
  }

  if (scale == 0.0f) {
    if (has_nonzero) {
      return SQ_ERR_SCALE;
    }
    if (bit_width == SQ_Q8_BITS) {
      for (i = 0; i < count; i++) {
        payload[i] = (uint8_t)state->signed_limit;
      }
    } else {
      size_t pairs = count / 2;
      uint8_t center = (uint8_t)(state->signed_limit & 0x0F);
      for (i = 0; i < pairs; i++) {
        payload[i] = (uint8_t)(center | (center << 4));
      }
      if (count % 2 == 1) {
        payload[pairs] = center;
      }
    }
    state->skip_encoding = 1;
  }
  return SQ_OK;
}

sq_status sq_encode_payload(uint8_t bit_width, const float *values,
                            size_t count, float scale, const uint32_t *words,
                            uint8_t *payload) {
  size_t i;
  sq_encode_payload_state state;
  int s;
  sq_status status;

  status = sq_encode_payload_prologue(bit_width, values, count, scale, words,
                                      payload, scan_scalar_encode_input, NULL,
                                      &state);
  if (status != SQ_OK) {
    return status;
  }
  if (state.skip_encoding) {
    return SQ_OK;
  }
  s = state.signed_limit;

  if (bit_width == SQ_Q8_BITS) {
    for (i = 0; i < count; i++) {
      float absolute_value = fabsf(values[i]);
      float scaled = (absolute_value / scale) * (float)s;
      float lower_float;
      float probability;
      int magnitude, signed_code;

      if (scaled > (float)s) {
        scaled = (float)s;
      }
      lower_float = floorf(scaled);
      probability = scaled - lower_float;
      magnitude = (int)lower_float + sq_bernoulli(words[i], probability);
      signed_code =
          signbit(values[i]) && magnitude != 0 ? -magnitude : magnitude;
      payload[i] = (uint8_t)(signed_code + s);
    }
  } else {
    size_t pairs = count / 2;
    for (i = 0; i < pairs; i++) {
      int sc0, sc1;
      float av0 = fabsf(values[2 * i]);
      float scaled0 = (av0 / scale) * (float)s;
      float lf0, prob0;
      int mag0;

      float av1 = fabsf(values[2 * i + 1]);
      float scaled1 = (av1 / scale) * (float)s;
      float lf1, prob1;
      int mag1;

      if (scaled0 > (float)s) {
        scaled0 = (float)s;
      }
      lf0 = floorf(scaled0);
      prob0 = scaled0 - lf0;
      mag0 = (int)lf0 + sq_bernoulli(words[2 * i], prob0);
      sc0 = signbit(values[2 * i]) && mag0 != 0 ? -mag0 : mag0;

      if (scaled1 > (float)s) {
        scaled1 = (float)s;
      }
      lf1 = floorf(scaled1);
      prob1 = scaled1 - lf1;
      mag1 = (int)lf1 + sq_bernoulli(words[2 * i + 1], prob1);
      sc1 = signbit(values[2 * i + 1]) && mag1 != 0 ? -mag1 : mag1;

      payload[i] = (uint8_t)(((sc0 + s) & 0x0F) | (((sc1 + s) & 0x0F) << 4));
    }
    if (count % 2 == 1) {
      float av = fabsf(values[count - 1]);
      float scaled = (av / scale) * (float)s;
      float lf, prob;
      int mag, sc;

      if (scaled > (float)s) {
        scaled = (float)s;
      }
      lf = floorf(scaled);
      prob = scaled - lf;
      mag = (int)lf + sq_bernoulli(words[count - 1], prob);
      sc = signbit(values[count - 1]) && mag != 0 ? -mag : mag;

      payload[pairs] = (uint8_t)((sc + s) & 0x0F);
    }
  }
  return SQ_OK;
}
