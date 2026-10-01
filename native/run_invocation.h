#ifndef SQ_RUN_INVOCATION_H
#define SQ_RUN_INVOCATION_H

#include <stdint.h>

static inline int sq_invocation_range_valid(uint64_t tensor_id,
                                            uint64_t base_invocation_id,
                                            uint64_t warmups, uint64_t reps,
                                            uint64_t step) {
  uint64_t total_runs;

  if (tensor_id > UINT32_MAX || base_invocation_id > UINT32_MAX ||
      warmups > UINT64_MAX - reps) {
    return 0;
  }
  total_runs = warmups + reps;
  if (total_runs == 0) {
    return 0;
  }
  return step == 0 ||
         total_runs - 1 <= ((uint64_t)UINT32_MAX - base_invocation_id) / step;
}

static inline uint64_t sq_run_invocation_id(uint64_t base_invocation_id,
                                            uint64_t warmups, uint64_t reps,
                                            uint64_t execution_index,
                                            uint64_t step) {
  const uint64_t offset = execution_index < warmups ? reps + execution_index
                                                    : execution_index - warmups;
  return base_invocation_id + offset * step;
}

#endif
