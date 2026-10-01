#ifndef SQ_BENCH_CLOCK_H
#define SQ_BENCH_CLOCK_H

#ifdef _WIN32
#include <windows.h>

typedef LARGE_INTEGER bench_clock_frequency;
static inline int bench_clock_init(bench_clock_frequency *frequency) {
  return QueryPerformanceFrequency(frequency) != 0;
}
static inline int bench_clock_now_ms(const bench_clock_frequency *frequency,
                                     double *milliseconds) {
  LARGE_INTEGER counter;
  if (QueryPerformanceCounter(&counter) == 0) {
    return 0;
  }
  *milliseconds =
      (double)counter.QuadPart * 1000.0 / (double)frequency->QuadPart;
  return 1;
}
#else
#include <time.h>

typedef int bench_clock_frequency;
static inline int bench_clock_init(bench_clock_frequency *frequency) {
  (void)frequency;
  return 1;
}
static inline int bench_clock_now_ms(const bench_clock_frequency *frequency,
                                     double *milliseconds) {
  struct timespec current;
  (void)frequency;
  if (clock_gettime(CLOCK_MONOTONIC, &current) != 0) {
    return 0;
  }
  *milliseconds =
      (double)current.tv_sec * 1000.0 + (double)current.tv_nsec / 1000000.0;
  return 1;
}
#endif

#endif
