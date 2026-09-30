#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rng_cpu.h"
#include "rng_cuda.h"
#include "test_alloc.h"
#include "test_check.h"

static int cuda_ok(int status, const char *what) {
  if (status != 0) {
    printf("FAIL %s: CUDA error %d\n", what, status);
    failures++;
    return 0;
  }
  return 1;
}

static int same_ctr(philox4x32_ctr_t a, philox4x32_ctr_t b) {
  return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] &&
         a.v[3] == b.v[3];
}

/* philox4x32 10-round rows from Random123 v1.14.0 tests/kat_vectors. */
static const uint32_t kat[3][10] = {
    {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
     0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
    {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0,
     0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
};

static void test_kat(void) {
  int row;

  for (row = 0; row < 3; row++) {
    philox4x32_ctr_t ctr, expected, gpu;
    philox4x32_key_t key;
    char what[64];
    int j;

    for (j = 0; j < 4; j++) {
      ctr.v[j] = kat[row][j];
      expected.v[j] = kat[row][6 + j];
    }
    key.v[0] = kat[row][4];
    key.v[1] = kat[row][5];

    (void)snprintf(what, sizeof what, "Philox4x32-10 KAT row %d on GPU", row);
    if (cuda_ok(sq_philox_raw_cuda(ctr, key, &gpu), what)) {
      check(same_ctr(gpu, expected), what);
    }
  }
}

static void test_geometry(void) {
  /* grid 0 picks the capped automatic grid; block 1 forces the cap. */
  static const int geometry[][2] = {{1, 0},   {32, 1},  {96, 13},
                                    {128, 7}, {256, 0}, {1024, 3}};
  const uint64_t n = 1000003; /* not a multiple of 4 */
  uint32_t *cpu = test_alloc(n, sizeof *cpu);
  uint32_t *gpu = test_alloc(n, sizeof *gpu);
  sq_rng_stream s;
  size_t g;

  sq_rng_stream_init(&s, 0xfeedfacecafebeefULL, 3, 42);
  sq_rng_words_cpu(&s, n, cpu);

  for (g = 0; g < sizeof geometry / sizeof geometry[0]; g++) {
    char what[96];
    (void)snprintf(what, sizeof what,
                   "GPU words match CPU for n=%llu, block=%d, grid=%d",
                   (unsigned long long)n, geometry[g][0], geometry[g][1]);
    memset(gpu, 0, n * sizeof *gpu);
    if (cuda_ok(sq_rng_words_cuda(&s, n, gpu, geometry[g][0], geometry[g][1]),
                what)) {
      check(memcmp(cpu, gpu, n * sizeof *cpu) == 0, what);
    }
  }
  free(cpu);
  free(gpu);
}

static void test_bernoulli(void) {
  static const uint32_t words[] = {0,          1,          0x7fffffff,
                                   0x80000000, 0xfffffeff, 0xffffff00,
                                   0xfffffffe, 0xffffffff};
  /* 0x1.fffffep-1f = 1 - 2^-24, the largest FP32 below 1. */
  static const float probs[] = {0.0f, 1.0f, 0x1.fffffep-1f, 0.5f};
  enum {
    NW = sizeof words / sizeof words[0],
    NP = sizeof probs / sizeof probs[0],
    N = NW * NP
  };
  uint32_t w[NW * NP];
  float p[NW * NP];
  uint8_t expected[NW * NP], cpu[NW * NP], gpu[NW * NP];
  int i, j, k = 0;

  for (j = 0; j < NP; j++) {
    for (i = 0; i < NW; i++, k++) {
      w[k] = words[i];
      p[k] = probs[j];
      if (j == 0) {
        expected[k] = 0;
      } else if (j == 1) {
        expected[k] = 1;
      } else if (j == 2) {
        expected[k] = words[i] < 0xffffff00u;
      } else {
        expected[k] = words[i] < 0x80000000u;
      }
    }
  }

  check(sq_bernoulli_threshold(0.0f) == 0, "threshold(0) = 0");
  check(sq_bernoulli_threshold(1.0f) == 0x100000000ULL, "threshold(1) = 2^32");

  for (i = 0; i < N; i++) {
    cpu[i] = (uint8_t)sq_bernoulli(w[i], p[i]);
  }
  check(memcmp(cpu, expected, sizeof expected) == 0,
        "CPU Bernoulli at p = 0, 1, 1-2^-24, 0.5");
  if (cuda_ok(sq_bernoulli_cuda(w, p, N, gpu), "GPU Bernoulli")) {
    check(memcmp(gpu, expected, sizeof expected) == 0,
          "GPU Bernoulli at p = 0, 1, 1-2^-24, 0.5");
  }
}

int main(void) {
  char name[256];
  int major, minor;

  if (cuda_ok(sq_cuda_device_info(name, sizeof name, &major, &minor),
              "device query")) {
    printf("# device: %s (compute capability %d.%d)\n", name, major, minor);
  }

  test_kat();
  test_geometry();
  test_bernoulli();

  printf("# %d failure(s)\n", failures);
  return failures ? 1 : 0;
}
