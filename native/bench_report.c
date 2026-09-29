#include "bench_report.h"

#include "codec.h"

#include <stdio.h>

static double sample_wall(const sq_bench_sample *sample) {
  return sample->wall_ms;
}
static double sample_k1(const sq_bench_sample *sample) { return sample->k1_ms; }
static double sample_k2(const sq_bench_sample *sample) { return sample->k2_ms; }
static double sample_k3(const sq_bench_sample *sample) { return sample->k3_ms; }
static double sample_h2d(const sq_bench_sample *sample) {
  return sample->h2d_ms;
}
static double sample_d2h(const sq_bench_sample *sample) {
  return sample->d2h_ms;
}
static double sample_cpu(const sq_bench_sample *sample) {
  return sample->cpu_ms;
}

typedef struct {
  unsigned int flag;
  const char *key;
  double (*field)(const sq_bench_sample *sample);
} column_row;

/* Written in this order after samples_ms. */
static const column_row column_table[] = {
    {SQ_BENCH_COLUMN_K1, "k1_ms", sample_k1},
    {SQ_BENCH_COLUMN_K2, "k2_ms", sample_k2},
    {SQ_BENCH_COLUMN_K3, "k3_ms", sample_k3},
    {SQ_BENCH_COLUMN_H2D, "h2d_ms", sample_h2d},
    {SQ_BENCH_COLUMN_D2H, "d2h_ms", sample_d2h},
    {SQ_BENCH_COLUMN_CPU, "cpu_ms", sample_cpu},
};

static void print_double_array(const sq_bench_sample *samples, uint64_t reps,
                               double (*field)(const sq_bench_sample *)) {
  uint64_t i;

  putchar('[');
  for (i = 0; i < reps; i++) {
    if (i != 0) {
      putchar(',');
    }
    printf("%.9f", field(&samples[i]));
  }
  putchar(']');
}

/* A step of 0 prints the base identifier for every run (resident-graph). */
static void print_invocation_ids(uint64_t base_invocation_id, uint64_t count,
                                 uint64_t offset, uint64_t step) {
  uint64_t i;

  putchar('[');
  for (i = 0; i < count; i++) {
    if (i != 0) {
      putchar(',');
    }
    printf("%llu",
           (unsigned long long)(base_invocation_id + (offset + i) * step));
  }
  putchar(']');
}

void sq_bench_print_json(const sq_bench_result *result) {
  const uint64_t step = result->invocation_id_step;
  size_t i;

  printf("{\"configuration\":{\"backend\":\"%s\",\"bits\":%u,"
         "\"count\":%llu,\"seed\":%llu,\"tensor_id\":%llu,"
         "\"invocation_id\":%llu,\"warmup\":%llu,\"reps\":%llu,"
         "\"repetition_invocation_ids\":",
         result->backend, (unsigned int)result->bit_width,
         (unsigned long long)result->count, (unsigned long long)result->seed,
         (unsigned long long)result->tensor_id,
         (unsigned long long)result->invocation_id,
         (unsigned long long)result->warmups, (unsigned long long)result->reps);
  print_invocation_ids(result->invocation_id, result->reps, 0, step);
  printf(",\"warmup_invocation_ids\":");
  print_invocation_ids(result->invocation_id, result->warmups, result->reps,
                       step);
  printf(",\"boundary\":\"%s\",\"transfer_policy\":\"%s\","
         "\"block_size\":%d,\"grid_size\":%d,",
         result->boundary, result->transfer_policy, result->block_size,
         result->grid_size);
  if (result->k1 != NULL) {
    printf("\"k1\":\"%s\",", result->k1);
  }
  /* The team size OpenMP grants a probe region; with dynamic teams off the
     timed regions get the same size. */
  if (result->team_size != 0) {
    printf("\"threads\":%d,", result->team_size);
  }
  printf("\"prescribed_scale\":");
  if (result->prescribed_scale_seen) {
    printf("%.9g", (double)result->prescribed_scale);
  } else {
    printf("null");
  }
  printf("},\"samples_ms\":");
  print_double_array(result->samples, result->reps, sample_wall);
  for (i = 0; i < sizeof column_table / sizeof column_table[0]; i++) {
    if ((result->columns & column_table[i].flag) != 0) {
      printf(",\"%s\":", column_table[i].key);
      print_double_array(result->samples, result->reps, column_table[i].field);
    }
  }
  if ((result->columns & SQ_BENCH_COLUMN_CAPTURE) != 0) {
    printf(",\"capture_and_instantiate_ms\":%.6f", result->capture_ms);
  }
  printf(",\"header_bytes\":%d,\"payload_bytes\":%llu}\n", SQ_HEADER_SIZE,
         (unsigned long long)result->payload_bytes);
}
