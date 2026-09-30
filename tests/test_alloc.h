#ifndef SQ_TEST_ALLOC_H
#define SQ_TEST_ALLOC_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Poison-filled allocation catches accidental reliance on zeroed test buffers.
 */
static inline void *test_alloc(size_t count, size_t size) {
  size_t bytes;
  void *pointer;

  if (count != 0 && size > SIZE_MAX / count) {
    (void)fprintf(stderr, "test allocation size overflow: %zu x %zu bytes\n",
                  count, size);
    exit(2);
  }
  bytes = count * size;
  pointer = malloc(bytes);
  /* malloc may return NULL for a zero-byte request without failing. */
  if (pointer == NULL && bytes != 0) {
    (void)fprintf(stderr, "test allocation of %zu x %zu bytes failed\n", count,
                  size);
    exit(2);
  }
  if (bytes != 0) {
    memset(pointer, 0xA5, bytes);
  }
  return pointer;
}

#endif
