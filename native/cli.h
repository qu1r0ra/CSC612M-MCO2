#ifndef SQ_CLI_H
#define SQ_CLI_H

#include <stddef.h>
#include <stdint.h>

#include "bench_types.h"
#include "sq_status.h"

#define SQ_MAX_CPU_THREADS 256

typedef enum {
  SQ_COMMAND_COMPRESS = 0,
  SQ_COMMAND_BENCH = 1,
  SQ_COMMAND_DECOMPRESS = 2,
  SQ_COMMAND_EXPECT = 3
} sq_command;

/* Bits of sq_options.seen: the options a command line named explicitly. */
enum {
  SQ_SEEN_SEED = 1 << 0,
  SQ_SEEN_SEEDS = 1 << 1,
  SQ_SEEN_SCALE = 1 << 2,
  SQ_SEEN_THREADS = 1 << 3,
  SQ_SEEN_BOUNDARY = 1 << 4,
  SQ_SEEN_POLICY = 1 << 5,
  SQ_SEEN_BLOCK_SIZE = 1 << 6,
  SQ_SEEN_GRID_SIZE = 1 << 7,
  SQ_SEEN_K1 = 1 << 8,
  SQ_SEEN_TIMINGS = 1 << 9
};

/* One options struct for every command. After sq_cli_validate, boundary and
   transfer_policy hold the resolved values, defaults included. */
typedef struct {
  const char *input_path;
  const char *output_path;
  const char *words_path;
  sq_backend backend;
  sq_boundary boundary;
  sq_transfer_policy transfer_policy;
  uint64_t seed;
  uint64_t seeds;
  uint64_t seed_start;
  uint64_t tensor_id;
  uint64_t invocation_id;
  uint64_t bits;
  uint64_t block_size;
  uint64_t grid_size;
  uint64_t threads;
  uint64_t warmups;
  uint64_t reps;
  float scale;
  int k1;
  unsigned int seen;
} sq_options;

/* One row per legal (backend, boundary) pair. The row also names the boundary
   on the wire and selects the stage columns of the bench report. */
typedef struct {
  sq_backend backend;
  sq_boundary boundary;
  const char *name;
  unsigned int columns;
  int transfers;    /* --transfer-policy applies */
  int needs_cuda;   /* absent from a CPU-only build */
  int is_default;   /* used when --boundary is absent */
  int named_on_cli; /* accepted after --boundary */
  uint64_t id_step; /* invocation-id increment per run */
} sq_legality;

/* Decimal digits only: no sign, no whitespace, no base prefix. */
int sq_parse_u64(const char *text, uint64_t *value);

sq_status sq_cli_parse(sq_command command, int argc, char **argv,
                       sq_options *options);
/* Checks combinations and required options, and resolves boundary and
   transfer_policy. cuda_available is 0 in a CPU-only build. */
sq_status sq_cli_validate(sq_command command, sq_options *options,
                          int cuda_available);

const sq_legality *sq_cli_legality(sq_backend backend, sq_boundary boundary);
const char *sq_backend_name(sq_backend backend);
const char *sq_transfer_policy_name(sq_transfer_policy policy);
const char *sq_k1_name(int variant);

/* Whole-file helpers; sizes of 2 GiB and above are supported. */
int sq_read_file(const char *path, uint8_t **bytes, size_t *size);
int sq_write_file(const char *path, const uint8_t *bytes, size_t size);

#endif
