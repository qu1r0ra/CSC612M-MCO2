#ifndef SQ_TEST_CHECK_H
#define SQ_TEST_CHECK_H

#include <stdio.h>

/* Shared pass/fail reporting for the C tests. Define TEST_CHECK_QUIET before
   including this header to print failures only. */
static int failures;
static int checks;

static inline void check(int condition, const char *description) {
  checks++;
#ifndef TEST_CHECK_QUIET
  printf("%s %s\n", condition ? "ok  " : "FAIL", description);
#else
  if (!condition) {
    printf("FAIL %s\n", description);
  }
#endif
  if (!condition) {
    failures++;
  }
}

#endif
