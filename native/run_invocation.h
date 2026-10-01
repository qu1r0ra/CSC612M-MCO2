#ifndef SQ_RUN_INVOCATION_H
#define SQ_RUN_INVOCATION_H

#include <stdint.h>

static inline uint64_t sq_run_invocation_id(uint64_t base_invocation_id,
                                            uint64_t warmups, uint64_t reps,
                                            uint64_t execution_index,
                                            uint64_t step) {
  const uint64_t offset = execution_index < warmups ? reps + execution_index
                                                    : execution_index - warmups;
  return base_invocation_id + offset * step;
}

#endif
