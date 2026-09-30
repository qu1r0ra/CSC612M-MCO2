#include <stdio.h>

#include "rng_cpu.h"
#include "test_check.h"

/* philox4x32 10-round rows from Random123 v1.14.0 tests/kat_vectors. */
static const uint32_t kat[3][10] = {
    {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
     0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
    {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0,
     0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
};

static int same_ctr(philox4x32_ctr_t a, philox4x32_ctr_t b) {
  return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] &&
         a.v[3] == b.v[3];
}

static void test_kat(void) {
  int row;

  for (row = 0; row < 3; row++) {
    philox4x32_ctr_t ctr, expected, actual;
    philox4x32_key_t key;
    char what[64];
    int j;

    for (j = 0; j < 4; j++) {
      ctr.v[j] = kat[row][j];
      expected.v[j] = kat[row][6 + j];
    }
    key.v[0] = kat[row][4];
    key.v[1] = kat[row][5];

    actual = philox4x32_R(SQ_PHILOX_ROUNDS, ctr, key);
    (void)snprintf(what, sizeof what, "Philox4x32-10 KAT row %d", row);
    check(same_ctr(actual, expected), what);
  }
}

static void test_mapping(void) {
  sq_rng_stream stream;
  philox4x32_key_t key;
  philox4x32_ctr_t ctr;
  uint32_t words[16];
  int i, ok = 1;

  sq_rng_stream_init(&stream, 0x0123456789abcdefULL, 7, 9);
  key = sq_philox_key(&stream);
  check(key.v[0] == 0x89abcdef && key.v[1] == 0x01234567,
        "key = {seed lo, seed hi}");

  ctr = sq_philox_ctr(&stream, 13 / 4);
  check(ctr.v[0] == 3 && ctr.v[1] == 0 && ctr.v[2] == 7 && ctr.v[3] == 9,
        "element 13 -> counter {group 3, 0, tensor, invocation}");

  ctr = sq_philox_ctr(&stream, 0x100000002ULL);
  check(ctr.v[0] == 2 && ctr.v[1] == 1, "64-bit group splits into lo/hi words");

  /* Element i takes lane i mod 4 of block floor(i/4), built by hand here. */
  sq_rng_words_cpu(&stream, 16, words);
  for (i = 0; i < 16; i++) {
    philox4x32_ctr_t counter = {{(uint32_t)(i / 4), 0, 7, 9}};
    philox4x32_ctr_t result = philox4x32_R(SQ_PHILOX_ROUNDS, counter, key);
    if (words[i] != result.v[i % 4]) {
      ok = 0;
    }
  }
  check(ok, "element i uses lane i mod 4 of group floor(i/4)");
}

static void test_overflow(void) {
  sq_rng_stream stream;

  check(sq_rng_stream_init(&stream, 1, UINT32_MAX, UINT32_MAX) == SQ_RNG_OK,
        "identifiers at UINT32_MAX accepted");
  check(sq_rng_stream_init(&stream, 1, (uint64_t)UINT32_MAX + 1, 0) ==
            SQ_RNG_ERR_ID_OVERFLOW,
        "tensor identifier above UINT32_MAX rejected");
  check(sq_rng_stream_init(&stream, 1, 0, (uint64_t)UINT32_MAX + 1) ==
            SQ_RNG_ERR_ID_OVERFLOW,
        "invocation identifier above UINT32_MAX rejected");
}

int main(void) {
  test_kat();
  test_mapping();
  test_overflow();

  printf("# %d failure(s)\n", failures);
  return failures ? 1 : 0;
}
