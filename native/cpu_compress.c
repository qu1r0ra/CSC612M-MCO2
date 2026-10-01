#include <stdlib.h>
#include <string.h>

#include "cpu_compress.h"
#include "quantizer.h"
#include "quantizer_avx2.h"
#include "rng_cpu.h"

void sq_cpu_compress_workspace_init(sq_cpu_compress_workspace *workspace) {
  if (workspace != NULL) {
    memset(workspace, 0, sizeof *workspace);
  }
}

sq_status
sq_cpu_compress_workspace_reserve(sq_cpu_compress_workspace *workspace,
                                  sq_backend backend, int threads,
                                  size_t count) {
  size_t partial_count;

  if (workspace == NULL ||
      (backend != SQ_BACKEND_CPU && backend != SQ_BACKEND_CPU_AVX2) ||
      (backend == SQ_BACKEND_CPU && threads != 0) ||
      (backend == SQ_BACKEND_CPU_AVX2 && threads < 1)) {
    return SQ_ERR_ARGUMENT;
  }
  if (count > SIZE_MAX / sizeof *workspace->generated_words) {
    return SQ_ERR_MEMORY;
  }
  partial_count = sq_scale_workspace_elements(count);
  if (partial_count == SIZE_MAX ||
      partial_count > SIZE_MAX / sizeof *workspace->scale_partials) {
    return SQ_ERR_MEMORY;
  }

  if (count > workspace->generated_word_capacity) {
    uint32_t *generated_words = (uint32_t *)realloc(
        workspace->generated_words, count * sizeof *workspace->generated_words);
    if (generated_words == NULL) {
      return SQ_ERR_MEMORY;
    }
    workspace->generated_words = generated_words;
    workspace->generated_word_capacity = count;
  }
  if (partial_count > workspace->scale_partial_capacity) {
    float *scale_partials =
        (float *)realloc(workspace->scale_partials,
                         partial_count * sizeof *workspace->scale_partials);
    if (scale_partials == NULL) {
      return SQ_ERR_MEMORY;
    }
    workspace->scale_partials = scale_partials;
    workspace->scale_partial_capacity = partial_count;
  }
  return SQ_OK;
}

void sq_cpu_compress_workspace_destroy(sq_cpu_compress_workspace *workspace) {
  if (workspace == NULL) {
    return;
  }
  free(workspace->generated_words);
  free(workspace->scale_partials);
  sq_cpu_compress_workspace_init(workspace);
}

static sq_status cpu_compute_scale(const float *values, size_t count,
                                   float *scale, float *partials,
                                   size_t partial_capacity, sq_backend backend,
                                   int threads) {
  if (backend == SQ_BACKEND_CPU) {
    return sq_compute_scale_with_workspace(values, count, scale, partials,
                                           partial_capacity);
  }
  return sq_avx2_compute_scale_with_workspace(values, count, scale, partials,
                                              partial_capacity, threads);
}

static void cpu_rng_words(const sq_rng_stream *stream, size_t count,
                          uint32_t *words, sq_backend backend, int threads) {
  if (backend == SQ_BACKEND_CPU) {
    sq_rng_words_cpu(stream, (uint64_t)count, words);
  } else {
    sq_avx2_rng_words(stream, (uint64_t)count, words, threads);
  }
}

static sq_status cpu_encode_payload(uint8_t bits, const float *values,
                                    size_t count, float scale,
                                    const uint32_t *words, uint8_t *payload,
                                    sq_backend backend, int threads) {
  if (backend == SQ_BACKEND_CPU) {
    return sq_encode_payload(bits, values, count, scale, words, payload);
  }
  return sq_avx2_encode_payload(bits, values, count, scale, words, payload,
                                threads);
}

sq_status sq_cpu_compress(sq_backend backend, uint8_t bit_width,
                          const float *values, size_t count, uint64_t seed,
                          uint64_t tensor_id, uint64_t invocation_id,
                          int prescribed_scale_seen, float prescribed_scale,
                          const uint32_t *prescribed_words, int threads,
                          sq_cpu_compress_workspace *workspace,
                          uint8_t *record) {
  uint32_t *generated_words;
  float *scale_partials;
  float scale;
  sq_rng_stream stream;
  sq_status status;

  generated_words = workspace->generated_words;
  scale_partials = workspace->scale_partials;

  if (prescribed_scale_seen) {
    scale = prescribed_scale;
  } else {
    status =
        cpu_compute_scale(values, count, &scale, scale_partials,
                          workspace->scale_partial_capacity, backend, threads);
    if (status != SQ_OK) {
      return status;
    }
  }
  if (prescribed_words == NULL) {
    if (sq_rng_stream_init(&stream, seed, tensor_id, invocation_id) !=
        SQ_RNG_OK) {
      return SQ_ERR_ID_OVERFLOW;
    }
    cpu_rng_words(&stream, count, generated_words, backend, threads);
  }
  status = sq_header_encode(bit_width, (uint64_t)count, scale, record);
  if (status != SQ_OK) {
    return status;
  }
  return cpu_encode_payload(bit_width, values, count, scale,
                            prescribed_words != NULL ? prescribed_words
                                                     : generated_words,
                            record + SQ_HEADER_SIZE, backend, threads);
}
