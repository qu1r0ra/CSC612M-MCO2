#ifndef SQ_BENCH_TYPES_H
#define SQ_BENCH_TYPES_H

#include <stddef.h>
#include <stdint.h>

/* Backend-neutral bench vocabulary shared by the CLI, the report writer and
   the CUDA driver. */

typedef enum {
  SQ_BACKEND_CPU = 0,
  SQ_BACKEND_CPU_AVX2 = 1,
  SQ_BACKEND_CUDA = 2
} sq_backend;

/* The first four values are the CUDA driver's boundaries; host-host is the
   CPU-only default. */
typedef enum {
  SQ_BOUNDARY_RESIDENT = 0,
  SQ_BOUNDARY_HOST_ORIGIN = 1,
  SQ_BOUNDARY_RESIDENT_GRAPH = 2,
  SQ_BOUNDARY_GPU_ORIGIN = 3,
  SQ_BOUNDARY_HOST_HOST = 4
} sq_boundary;

/* Host buffers that a timed transfer touches: malloc or cudaHostAlloc. */
typedef enum {
  SQ_TRANSFER_PAGEABLE = 0,
  SQ_TRANSFER_PINNED = 1,
  SQ_TRANSFER_NONE = 2
} sq_transfer_policy;

typedef struct {
  double wall_ms;
  double k1_ms;
  double k2_ms;
  double k3_ms;
  double h2d_ms;
  double d2h_ms;
  double cpu_ms;
} sq_bench_sample;

/* Stage columns a bench report carries beside samples_ms, in the order they
   are written. */
enum {
  SQ_BENCH_COLUMN_K1 = 1 << 0,
  SQ_BENCH_COLUMN_K2 = 1 << 1,
  SQ_BENCH_COLUMN_K3 = 1 << 2,
  SQ_BENCH_COLUMN_H2D = 1 << 3,
  SQ_BENCH_COLUMN_D2H = 1 << 4,
  SQ_BENCH_COLUMN_CPU = 1 << 5,
  SQ_BENCH_COLUMN_CAPTURE = 1 << 6
};

/* Everything print_bench_json needs; string fields are the wire names. */
typedef struct {
  const char *backend;
  const char *boundary;
  const char *transfer_policy;
  const char *k1; /* NULL when the backend has no K1 variant */
  unsigned int columns;
  uint64_t invocation_id_step;
  uint8_t bit_width;
  size_t count;
  uint64_t seed;
  uint64_t tensor_id;
  uint64_t invocation_id;
  uint64_t warmups;
  uint64_t reps;
  int prescribed_scale_seen;
  float prescribed_scale;
  int block_size;
  int grid_size;
  int team_size;
  size_t payload_bytes;
  const sq_bench_sample *samples;
  double capture_ms;
} sq_bench_result;

#endif
