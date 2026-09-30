/* Differential test: every AVX2 stage against its scalar counterpart, bit for
 * bit. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quantizer.h"
#include "quantizer_avx2.h"
#include "rng_cpu.h"
#include "test_alloc.h"
#define TEST_CHECK_QUIET
#include "test_check.h"

static void check_at(int condition, const char *description, size_t count,
                     int threads) {
  char what[160];
  if (!condition) {
    (void)snprintf(what, sizeof what, "%s (count %zu, threads %d)", description,
                   count, threads);
    check(0, what);
  } else {
    check(1, description);
  }
}

static uint64_t lcg_state = 0x9E3779B97F4A7C15ull;

static uint32_t next_u32(void) {
  lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(lcg_state >> 32);
}

/* Mixed magnitudes and signs, with zeros, -0 and subnormals sprinkled in. */
static void fill(float *values, size_t count, int kind) {
  size_t i;

  for (i = 0; i < count; i++) {
    uint32_t r = next_u32();
    float v = ((float)(r >> 8) / 16777216.0f) * 2.0f - 1.0f;

    if (kind == 1) {
      v *= ldexpf(1.0f, (int)(r % 60) - 30);
    } else if (kind == 2) {
      v = ldexpf(v, -140);
    }
    if (r % 97 == 0) {
      v = 0.0f;
    } else if (r % 89 == 0) {
      v = -0.0f;
    }
    values[i] = v;
  }
}

static int same_float(float a, float b) {
  uint32_t a_bits, b_bits;

  memcpy(&a_bits, &a, sizeof a_bits);
  memcpy(&b_bits, &b, sizeof b_bits);
  return a_bits == b_bits;
}

static void compare_scale(const float *values, size_t count, int threads,
                          const char *what) {
  size_t capacity = sq_scale_workspace_elements(count);
  float *p0 = capacity ? (float *)test_alloc(capacity, sizeof *p0) : NULL;
  float *p1 = capacity ? (float *)test_alloc(capacity, sizeof *p1) : NULL;
  float s0 = -1.0f, s1 = -2.0f;
  sq_status st0 =
      sq_compute_scale_with_workspace(values, count, &s0, p0, capacity);
  sq_status st1 = sq_avx2_compute_scale_with_workspace(values, count, &s1, p1,
                                                       capacity, threads);

  check_at(st0 == st1 && same_float(s0, s1), what, count, threads);
  free(p0);
  free(p1);
}

static void compare_encode(uint8_t bits, const float *values, size_t count,
                           float scale, const uint32_t *words, int threads,
                           const char *what) {
  size_t bytes = sq_payload_size(bits, count);
  uint8_t *e0 = (uint8_t *)test_alloc(bytes + 1, 1),
          *e1 = (uint8_t *)test_alloc(bytes + 1, 1);
  sq_status st0, st1;

  memset(e0, 0xA5, bytes + 1);
  memset(e1, 0xA5, bytes + 1);
  st0 = sq_encode_payload(bits, values, count, scale, words, e0);
  st1 = sq_avx2_encode_payload(bits, values, count, scale, words, e1, threads);
  check_at(st0 == st1 && memcmp(e0, e1, bytes + 1) == 0, what, count, threads);
  free(e0);
  free(e1);
}

static void compare_encode_status(uint8_t bits, const float *values,
                                  size_t count, float scale,
                                  const uint32_t *words, uint8_t *payload,
                                  int threads, sq_status expected,
                                  const char *what) {
  sq_status st0 = sq_encode_payload(bits, values, count, scale, words, payload);
  sq_status st1 = sq_avx2_encode_payload(bits, values, count, scale, words,
                                         payload, threads);
  check_at(st0 == expected && st1 == expected, what, count, threads);
}

static void compare_all(const float *values, size_t count, int threads,
                        uint64_t seed) {
  sq_rng_stream stream;
  uint32_t *w0 = (uint32_t *)test_alloc(count + 1, sizeof *w0);
  uint32_t *w1 = (uint32_t *)test_alloc(count + 1, sizeof *w1);
  float scale = 0.0f;

  sq_rng_stream_init(&stream, seed, 7, 0xFFFFFFFFu);
  sq_rng_words_cpu(&stream, count, w0);
  w0[count] = w1[count] = 0xDEADBEEFu;
  sq_avx2_rng_words(&stream, count, w1, threads);
  check_at(memcmp(w0, w1, (count + 1) * sizeof *w0) == 0, "rng words", count,
           threads);

  compare_scale(values, count, threads, "scale");
  if (sq_compute_scale(values, count, &scale) == SQ_OK) {
    compare_encode(SQ_Q8_BITS, values, count, scale, w0, threads,
                   "8-bit payload");
    compare_encode(SQ_Q4_BITS, values, count, scale, w0, threads,
                   "4-bit payload");
  }
  compare_encode(SQ_Q8_BITS, values, count, 1e-30f, w0, threads,
                 "8-bit clamped payload");
  compare_encode(SQ_Q4_BITS, values, count, 1e-30f, w0, threads,
                 "4-bit clamped payload");
  compare_encode(SQ_Q8_BITS, values, count, 3.0f, w0, threads,
                 "8-bit prescribed payload");
  free(w0);
  free(w1);
}

int main(void) {
  static const size_t counts[] = {0,
                                  1,
                                  2,
                                  3,
                                  4,
                                  5,
                                  7,
                                  255,
                                  256,
                                  257,
                                  511,
                                  1000,
                                  1287,
                                  4096,
                                  65549,
                                  99999,
                                  (1u << 20) + 13};
  static const int team[] = {1, 2, 3, 7, 16};
  const float probes[] = {NAN, INFINITY, -INFINITY};
  const size_t big = (1u << 20) + 13;
  float *values = (float *)test_alloc(big, sizeof *values);
  uint32_t *words = (uint32_t *)test_alloc(big, sizeof *words);
  size_t c, i;
  int t, k, kind;

  for (c = 0; c < sizeof counts / sizeof counts[0]; c++) {
    for (kind = 0; kind < 3; kind++) {
      fill(values, counts[c], kind);
      for (t = 0; t < (int)(sizeof team / sizeof team[0]); t++) {
        compare_all(values, counts[c], team[t], 0x0123456789ABCDEFull + c);
      }
    }
  }

  for (i = 0; i < big; i++) {
    words[i] = next_u32();
  }

  /* All zeros, including -0: zero scale, and a nonzero scale is rejected. */
  for (i = 0; i < 1000; i++) {
    values[i] = i % 3 == 0 ? -0.0f : 0.0f;
  }
  for (t = 0; t < (int)(sizeof team / sizeof team[0]); t++) {
    compare_scale(values, 1000, team[t], "all-zero scale");
    compare_encode(SQ_Q8_BITS, values, 1000, 0.0f, words, team[t],
                   "all-zero 8-bit");
    compare_encode(SQ_Q4_BITS, values, 999, 0.0f, words, team[t],
                   "all-zero 4-bit odd");
    compare_encode(SQ_Q8_BITS, values, 1000, 1.0f, words, team[t],
                   "all-zero nonzero scale");
  }

  /* Non-finite values at the head, middle and tail block. */
  for (k = 0; k < 3; k++) {
    static const size_t at[] = {0, 700, 1286};
    size_t p;

    for (p = 0; p < 3; p++) {
      fill(values, 1287, 0);
      values[at[p]] = probes[k];
      for (t = 0; t < (int)(sizeof team / sizeof team[0]); t++) {
        compare_scale(values, 1287, team[t], "non-finite scale status");
        compare_encode(SQ_Q8_BITS, values, 1287, 1.0f, words, team[t],
                       "non-finite encode status");
      }
    }
  }

  /* Status paths that return before any work. */
  fill(values, 300, 0);
  for (t = 0; t < (int)(sizeof team / sizeof team[0]); t++) {
    float nonfinite = INFINITY;
    uint32_t word = 0;
    uint8_t payload = 0;

    compare_encode_status(3, NULL, 1, NAN, NULL, NULL, team[t],
                          SQ_ERR_BIT_WIDTH, "bit width precedes arguments");
    compare_encode_status(SQ_Q8_BITS, NULL, 1, -1.0f, NULL, NULL, team[t],
                          SQ_ERR_ARGUMENT, "arguments precede scale");
    compare_encode_status(SQ_Q8_BITS, &nonfinite, 1, -1.0f, &word, &payload,
                          team[t], SQ_ERR_SCALE,
                          "scale precedes non-finite input");
    compare_encode_status(SQ_Q8_BITS, &nonfinite, 1, 0.0f, &word, &payload,
                          team[t], SQ_ERR_NONFINITE,
                          "non-finite input precedes zero-scale fill");
    values[0] = FLT_MAX;
    values[1] = FLT_MAX;
    compare_scale(values, 300, team[t], "scale overflow");
    compare_encode(SQ_Q8_BITS, values, 300, -1.0f, words, team[t],
                   "negative scale");
    compare_encode(SQ_Q8_BITS, values, 300, NAN, words, team[t], "NaN scale");
    compare_encode(SQ_Q8_BITS, values, 300, INFINITY, words, team[t],
                   "infinite scale");
    compare_encode(SQ_Q8_BITS, values, 0, 1.0f, words, team[t],
                   "empty with scale");
    compare_encode(SQ_Q8_BITS, values, 0, 0.0f, words, team[t], "empty");
    compare_encode(SQ_Q8_BITS, values, 300, 0.0f, words, team[t],
                   "zero scale, nonzero");
    compare_encode(3, values, 300, 1.0f, words, team[t], "bit width");
    compare_encode(SQ_Q8_BITS, values, 300, FLT_MIN / 4.0f, words, team[t],
                   "subnormal scale");
  }

  /* Extreme words: every threshold boundary gets hit by 0 and UINT32_MAX. */
  fill(values, 65549, 1);
  for (i = 0; i < 65549; i++) {
    words[i] = i % 2 != 0 ? UINT32_MAX : 0;
  }
  for (t = 0; t < (int)(sizeof team / sizeof team[0]); t++) {
    float scale;
    sq_compute_scale(values, 65549, &scale);
    compare_encode(SQ_Q8_BITS, values, 65549, scale, words, team[t],
                   "extreme words 8-bit");
    compare_encode(SQ_Q4_BITS, values, 65549, scale, words, team[t],
                   "extreme words 4-bit");
  }

  for (t = 0; t < (int)(sizeof team / sizeof team[0]); t++) {
    check_at(sq_avx2_team_size(team[t]) == team[t], "team size", 0, team[t]);
  }

  free(values);
  free(words);
  printf("%d checks, %d failures\n", checks, failures);
  return failures != 0;
}
