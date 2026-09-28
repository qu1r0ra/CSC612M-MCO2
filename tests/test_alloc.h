#ifndef SQ_TEST_ALLOC_H
#define SQ_TEST_ALLOC_H

#include <stdio.h>
#include <stdlib.h>

/* Zeroed allocation that ends the run on failure, so no test needs a NULL
   branch of its own. */
static inline void *test_alloc(size_t count, size_t size) {
  void *pointer = calloc(count, size);
  /* calloc may return NULL for a zero-byte request without failing. */
  if (pointer == NULL && count != 0 && size != 0) {
    (void)fprintf(stderr, "test allocation of %zu x %zu bytes failed\n", count,
                  size);
    exit(2);
  }
  return pointer;
}

#endif
