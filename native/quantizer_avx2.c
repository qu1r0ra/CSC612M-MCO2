#include "quantizer_avx2.h"

#include <math.h>
#include <omp.h>
#include <string.h>

#include "quantizer.h"

/*
 * The loops are written for the auto-vectorizer. Loops tagged "avx2-hot" must
 * appear as vectorized in the compiler report (see `just vec-report`).
 * Untagged loops drive tagged ones chunk by chunk, or are scalar: the Philox
 * word interleave, the 4-bit nibble pack, and the per-block bit reversal of
 * the scale tree.
 * Every float operation matches quantizer.c one for one; contraction stays off.
 */
#ifdef _MSC_VER
#pragma fp_contract(off)
#endif

#define BLOCK SQ_SCALE_BLOCK_SIZE
#define PHILOX_CHUNK 64
#define MAGNITUDE_MASK 0x7FFFFFFFu
#define NONFINITE_BITS 0x7F800000u

/* Contiguous share of n items for one thread; shares differ by at most one. */
static void split(size_t n, int thread, int team, size_t *begin, size_t *end) {
  size_t base = n / (size_t)team, extra = n % (size_t)team;
  size_t t = (size_t)thread;

  *begin = t * base + (t < extra ? t : extra);
  *end = *begin + base + (t < extra ? 1 : 0);
}

/* Element range of whole 256-element units, so 4-bit pairs never straddle
 * threads. */
static void unit_range(size_t count, int thread, int team, size_t *begin,
                       size_t *end) {
  size_t units = count / BLOCK + (count % BLOCK != 0);

  split(units, thread, team, begin, end);
  *begin *= BLOCK;
  *end *= BLOCK;
  if (*end > count) {
    *end = count;
  }
}

static size_t reverse_bits(size_t x, int bits) {
  size_t r = 0;
  int i;

  for (i = 0; i < bits; i++) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

int sq_avx2_team_size(int threads) {
  int team = 0;

  omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
  {
#pragma omp master
    team = omp_get_num_threads();
  }
  return team;
}

/* Largest |x| bit pattern: >= NONFINITE_BITS means inf or NaN, 0 means all
 * zeros. */
static uint32_t max_magnitude_bits(const float *values, size_t count,
                                   int threads) {
  const uint32_t *bits = (const uint32_t *)values;
  uint32_t result = 0;

  omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
  {
    size_t begin, end, i;
    uint32_t local = 0;

    unit_range(count, omp_get_thread_num(), omp_get_num_threads(), &begin,
               &end);
    for (i = begin; i < end; i++) { /* avx2-hot */
      uint32_t m = bits[i] & MAGNITUDE_MASK;
      local = m > local ? m : local;
    }
#pragma omp critical
    {
      if (local > result) {
        result = local;
      }
    }
  }
  return result;
}

/*
 * terms[j] holds the term of element rev[j]. Pairing j with j + half on
 * bit-reversed storage adds the same operands at every level as the adjacent
 * tree terms[i] = terms[2i] + terms[2i+1] in quantizer.c.
 */
static void gather_terms(const float *restrict block,
                         const int32_t *restrict rev, float *restrict terms,
                         float max_abs) {
  int j;

  for (j = 0; j < BLOCK; j++) { /* avx2-hot */
    float ratio = fabsf(block[rev[j]]) / max_abs;
    terms[j] = ratio * ratio;
  }
}

/* lo and hi are the two disjoint halves of one level, so the vectorizer can see
 * it. */
static void add_halves(float *restrict lo, const float *restrict hi,
                       size_t half) {
  size_t j;

  for (j = 0; j < half; j++) { /* avx2-hot */
    lo[j] = lo[j] + hi[j];
  }
}

static void halves_tree(float *terms, size_t n) {
  size_t half;

  for (half = n / 2; half != 0; half /= 2) {
    add_halves(terms, terms + half, half);
  }
}

static float block_sum(const float *values, size_t count, size_t start,
                       float max_abs, const int32_t *rev) {
  float tail[BLOCK], terms[BLOCK];
  const float *block = values + start;
  size_t n = count - start;

  if (n < BLOCK) {
    memset(tail, 0, sizeof tail);
    memcpy(tail, block, n * sizeof *tail);
    block = tail;
  }
  gather_terms(block, rev, terms, max_abs);
  halves_tree(terms, BLOCK);
  return terms[0];
}

sq_status sq_avx2_compute_scale_with_workspace(const float *values,
                                               size_t count, float *scale,
                                               float *partials,
                                               size_t partial_capacity,
                                               int threads) {
  int32_t rev[BLOCK];
  uint32_t max_bits;
  float max_abs;
  size_t block_count, padded_count, i;
  int padded_bits = 0;

  if (scale == NULL || (count != 0 && values == NULL)) {
    return SQ_ERR_ARGUMENT;
  }
  *scale = 0.0f;
  if (count == 0) {
    return SQ_OK;
  }

  max_bits = max_magnitude_bits(values, count, threads);
  if (max_bits >= NONFINITE_BITS) {
    return SQ_ERR_NONFINITE;
  }
  if (max_bits == 0) {
    return SQ_OK;
  }
  memcpy(&max_abs, &max_bits, sizeof max_abs);

  padded_count = sq_scale_workspace_elements(count);
  if (padded_count == SIZE_MAX) {
    return SQ_ERR_MEMORY;
  }
  block_count = count / BLOCK + (count % BLOCK != 0);
  if (padded_count > SIZE_MAX / sizeof *partials) {
    return SQ_ERR_MEMORY;
  }
  if (partials == NULL || partial_capacity < padded_count) {
    return SQ_ERR_ARGUMENT;
  }
  memset(partials, 0, padded_count * sizeof *partials);
  while (((size_t)1 << padded_bits) < padded_count) {
    padded_bits++;
  }
  for (i = 0; i < BLOCK; i++) {
    rev[i] = (int32_t)reverse_bits(i, 8);
  }

  omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
  {
    size_t begin, end, b;

    split(block_count, omp_get_thread_num(), omp_get_num_threads(), &begin,
          &end);
    for (b = begin; b < end; b++) {
      partials[reverse_bits(b, padded_bits)] =
          block_sum(values, count, b * BLOCK, max_abs, rev);
    }
  }
  halves_tree(partials, padded_count);

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

/* Philox4x32-10 for PHILOX_CHUNK consecutive groups, one lane per group. */
static void philox_chunk(uint64_t first_group, uint32_t k0, uint32_t k1,
                         uint32_t tensor, uint32_t invocation,
                         uint32_t *restrict out) {
  uint32_t c0[PHILOX_CHUNK], c1[PHILOX_CHUNK], c2[PHILOX_CHUNK],
      c3[PHILOX_CHUNK];
  int j, r;

  for (j = 0; j < PHILOX_CHUNK; j++) { /* avx2-hot */
    uint64_t group = first_group + (uint64_t)j;
    c0[j] = (uint32_t)group;
    c1[j] = (uint32_t)(group >> 32);
    c2[j] = tensor;
    c3[j] = invocation;
  }
  for (r = 0; r < SQ_PHILOX_ROUNDS; r++) {
    uint32_t ka = k0 + (uint32_t)r * 0x9E3779B9u;
    uint32_t kb = k1 + (uint32_t)r * 0xBB67AE85u;

    for (j = 0; j < PHILOX_CHUNK; j++) { /* avx2-hot */
      uint64_t p0 = (uint64_t)0xD2511F53u * c0[j];
      uint64_t p1 = (uint64_t)0xCD9E8D57u * c2[j];
      uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1[j] ^ ka;
      uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3[j] ^ kb;

      c1[j] = (uint32_t)p1;
      c3[j] = (uint32_t)p0;
      c0[j] = n0;
      c2[j] = n2;
    }
  }
  for (j = 0; j < PHILOX_CHUNK; j++) {
    out[4 * j] = c0[j];
    out[4 * j + 1] = c1[j];
    out[4 * j + 2] = c2[j];
    out[4 * j + 3] = c3[j];
  }
}

void sq_avx2_rng_words(const sq_rng_stream *s, uint64_t n, uint32_t *out,
                       int threads) {
  const uint64_t chunk_words = 4 * PHILOX_CHUNK;
  const size_t chunks = (size_t)((n + chunk_words - 1) / chunk_words);
  const uint32_t k0 = (uint32_t)s->seed, k1 = (uint32_t)(s->seed >> 32);
  const uint32_t tensor = s->tensor_id, invocation = s->invocation_id;

  omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
  {
    uint32_t tail[4 * PHILOX_CHUNK];
    size_t begin, end, c;

    split(chunks, omp_get_thread_num(), omp_get_num_threads(), &begin, &end);
    for (c = begin; c < end; c++) {
      uint64_t first_word = (uint64_t)c * chunk_words;

      if (first_word + chunk_words <= n) {
        philox_chunk((uint64_t)c * PHILOX_CHUNK, k0, k1, tensor, invocation,
                     out + first_word);
      } else {
        philox_chunk((uint64_t)c * PHILOX_CHUNK, k0, k1, tensor, invocation,
                     tail);
        memcpy(out + first_word, tail, (size_t)(n - first_word) * sizeof *tail);
      }
    }
  }
}

/*
 * lower = floor(scaled) and threshold = floor(p * 2^32) for p = scaled - lower,
 * as in sq_bernoulli_threshold. p * 2^32 can exceed INT32_MAX, so it is
 * truncated in two halves: q = trunc(t / 2), then t - 2q is 0 or 1, exactly.
 */
static void split_scaled(const float *restrict x, size_t n, float scale,
                         float limit, int32_t *restrict lower,
                         uint32_t *restrict threshold) {
  size_t k;

  for (k = 0; k < n; k++) { /* avx2-hot */
    float scaled = (fabsf(x[k]) / scale) * limit;
    float t;
    int32_t whole, q;

    scaled = scaled > limit ? limit : scaled;
    whole = (int32_t)scaled;
    t = (scaled - (float)whole) * 4294967296.0f;
    q = (int32_t)(t * 0.5f);
    lower[k] = whole;
    threshold[k] = (uint32_t)q * 2u + (uint32_t)(int32_t)(t - 2.0f * (float)q);
  }
}

/* Bernoulli draw, then the sign; (m ^ -neg) + neg is -m when neg is set, and -0
 * is 0. */
static void combine_codes(const uint32_t *restrict words,
                          const uint32_t *restrict bits,
                          const int32_t *restrict lower,
                          const uint32_t *restrict threshold, size_t n,
                          uint32_t s, uint8_t *restrict codes) {
  size_t k;

  for (k = 0; k < n; k++) { /* avx2-hot */
    uint32_t below =
        (uint32_t)(((uint64_t)words[k] - (uint64_t)threshold[k]) >> 63);
    uint32_t magnitude = (uint32_t)lower[k] + below;
    uint32_t negative = bits[k] >> 31;

    codes[k] = (uint8_t)(((magnitude ^ (0u - negative)) + negative) + s);
  }
}

static void pack_nibbles(const uint8_t *restrict codes, size_t n,
                         uint8_t *restrict out) {
  size_t pairs = n / 2, k;

  for (k = 0; k < pairs; k++) {
    out[k] =
        (uint8_t)((codes[2 * k] & 0x0F) | ((codes[2 * k + 1] & 0x0F) << 4));
  }
  if (n % 2 == 1) {
    out[pairs] = (uint8_t)(codes[n - 1] & 0x0F);
  }
}

static void encode_range(uint8_t bit_width, const float *values, size_t begin,
                         size_t end, float scale, uint32_t s,
                         const uint32_t *words, uint8_t *payload) {
  int32_t lower[BLOCK];
  uint32_t threshold[BLOCK];
  uint8_t codes[BLOCK];
  size_t base;

  for (base = begin; base < end; base += BLOCK) {
    size_t n = end - base < BLOCK ? end - base : BLOCK;

    split_scaled(values + base, n, scale, (float)s, lower, threshold);
    if (bit_width == SQ_Q8_BITS) {
      combine_codes(words + base, (const uint32_t *)(values + base), lower,
                    threshold, n, s, payload + base);
    } else {
      combine_codes(words + base, (const uint32_t *)(values + base), lower,
                    threshold, n, s, codes);
      pack_nibbles(codes, n, payload + base / 2);
    }
  }
}

sq_status sq_avx2_encode_payload(uint8_t bit_width, const float *values,
                                 size_t count, float scale,
                                 const uint32_t *words, uint8_t *payload,
                                 int threads) {
  uint32_t s, max_bits = 0;
  int has_nonzero;

  if (bit_width != SQ_Q4_BITS && bit_width != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  s = (bit_width == SQ_Q4_BITS) ? SQ_Q4_SIGNED_LIMIT : SQ_Q8_SIGNED_LIMIT;

  if (count != 0 && (values == NULL || words == NULL || payload == NULL)) {
    return SQ_ERR_ARGUMENT;
  }
  if (!isfinite(scale) || scale < 0.0f) {
    return SQ_ERR_SCALE;
  }
  if (count == 0 && scale != 0.0f) {
    return SQ_ERR_SCALE;
  }
  if (count != 0) {
    max_bits = max_magnitude_bits(values, count, threads);
  }
  if (max_bits >= NONFINITE_BITS) {
    return SQ_ERR_NONFINITE;
  }
  has_nonzero = max_bits != 0;
  if (!has_nonzero && scale != 0.0f) {
    return SQ_ERR_SCALE;
  }

  if (scale == 0.0f) {
    if (has_nonzero) {
      return SQ_ERR_SCALE;
    }
    if (count == 0) {
      return SQ_OK;
    }
    if (bit_width == SQ_Q8_BITS) {
      memset(payload, (int)s, count);
    } else {
      memset(payload, (int)((s & 0x0F) | ((s & 0x0F) << 4)), count / 2);
      if (count % 2 == 1) {
        payload[count / 2] = (uint8_t)(s & 0x0F);
      }
    }
    return SQ_OK;
  }

  omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
  {
    size_t begin, end;

    unit_range(count, omp_get_thread_num(), omp_get_num_threads(), &begin,
               &end);
    encode_range(bit_width, values, begin, end, scale, s, words, payload);
  }
  return SQ_OK;
}
