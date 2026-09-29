#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "cli.h"

#include "codec.h"
#include "cuda_limits.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/types.h>
#endif

#define SQ_READ_CHUNK ((size_t)1 << 30)

#define COMMAND_BIT(command) (1u << (command))
#define ALL_COMMANDS                                                           \
  (COMMAND_BIT(SQ_COMMAND_COMPRESS) | COMMAND_BIT(SQ_COMMAND_BENCH) |          \
   COMMAND_BIT(SQ_COMMAND_DECOMPRESS) | COMMAND_BIT(SQ_COMMAND_EXPECT))
#define TENSOR_COMMANDS                                                        \
  (COMMAND_BIT(SQ_COMMAND_COMPRESS) | COMMAND_BIT(SQ_COMMAND_BENCH) |          \
   COMMAND_BIT(SQ_COMMAND_EXPECT))
#define ENCODE_COMMANDS                                                        \
  (COMMAND_BIT(SQ_COMMAND_COMPRESS) | COMMAND_BIT(SQ_COMMAND_BENCH))

#define COUNT_OF(array) (sizeof(array) / sizeof((array)[0]))

#define KERNEL_COLUMNS                                                         \
  (SQ_BENCH_COLUMN_K1 | SQ_BENCH_COLUMN_K2 | SQ_BENCH_COLUMN_K3)

/* Add a boundary here and in sq_boundary; nothing else in the CLI or the
   report changes. */
static const sq_legality legality_table[] = {
    {.backend = SQ_BACKEND_CPU,
     .boundary = SQ_BOUNDARY_HOST_HOST,
     .name = "host-host",
     .columns = 0,
     .transfers = 0,
     .needs_cuda = 0,
     .is_default = 1,
     .named_on_cli = 0,
     .id_step = 1},
    {.backend = SQ_BACKEND_CPU,
     .boundary = SQ_BOUNDARY_GPU_ORIGIN,
     .name = "gpu-origin",
     .columns = SQ_BENCH_COLUMN_D2H | SQ_BENCH_COLUMN_CPU,
     .transfers = 1,
     .needs_cuda = 1,
     .is_default = 0,
     .named_on_cli = 1,
     .id_step = 1},
    {.backend = SQ_BACKEND_CPU_AVX2,
     .boundary = SQ_BOUNDARY_HOST_HOST,
     .name = "host-host",
     .columns = 0,
     .transfers = 0,
     .needs_cuda = 0,
     .is_default = 1,
     .named_on_cli = 0,
     .id_step = 1},
    {.backend = SQ_BACKEND_CUDA,
     .boundary = SQ_BOUNDARY_RESIDENT,
     .name = "resident",
     .columns = KERNEL_COLUMNS,
     .transfers = 0,
     .needs_cuda = 1,
     .is_default = 0,
     .named_on_cli = 1,
     .id_step = 1},
    {.backend = SQ_BACKEND_CUDA,
     .boundary = SQ_BOUNDARY_HOST_ORIGIN,
     .name = "host-origin",
     .columns = KERNEL_COLUMNS | SQ_BENCH_COLUMN_H2D | SQ_BENCH_COLUMN_D2H,
     .transfers = 1,
     .needs_cuda = 1,
     .is_default = 1,
     .named_on_cli = 1,
     .id_step = 1},
    {.backend = SQ_BACKEND_CUDA,
     .boundary = SQ_BOUNDARY_RESIDENT_GRAPH,
     .name = "resident-graph",
     .columns = SQ_BENCH_COLUMN_CAPTURE,
     .transfers = 0,
     .needs_cuda = 1,
     .is_default = 0,
     .named_on_cli = 1,
     .id_step = 0},
    {.backend = SQ_BACKEND_CUDA,
     .boundary = SQ_BOUNDARY_GPU_ORIGIN,
     .name = "gpu-origin",
     .columns = KERNEL_COLUMNS | SQ_BENCH_COLUMN_D2H,
     .transfers = 1,
     .needs_cuda = 1,
     .is_default = 0,
     .named_on_cli = 1,
     .id_step = 1},
};

typedef struct {
  const char *name;
  sq_backend backend;
  unsigned int commands;
  int is_cuda;
} backend_row;

static const backend_row backend_table[] = {
    {"cpu", SQ_BACKEND_CPU, ALL_COMMANDS, 0},
    {"cpu-avx2", SQ_BACKEND_CPU_AVX2, ENCODE_COMMANDS, 0},
    {"cuda", SQ_BACKEND_CUDA, ALL_COMMANDS, 1},
};

typedef struct {
  const char *name;
  sq_transfer_policy policy;
  int named_on_cli;
} policy_row;

static const policy_row policy_table[] = {
    {"pageable", SQ_TRANSFER_PAGEABLE, 1},
    {"pinned", SQ_TRANSFER_PINNED, 1},
    {"none", SQ_TRANSFER_NONE, 0},
};

typedef enum {
  OPT_INPUT,
  OPT_OUTPUT,
  OPT_WORDS,
  OPT_SEED,
  OPT_SEEDS,
  OPT_SEED_START,
  OPT_TENSOR_ID,
  OPT_INVOCATION_ID,
  OPT_BITS,
  OPT_SCALE,
  OPT_BACKEND,
  OPT_THREADS,
  OPT_BOUNDARY,
  OPT_POLICY,
  OPT_WARMUP,
  OPT_REPS,
  OPT_BLOCK_SIZE,
  OPT_GRID_SIZE,
  OPT_K1,
  OPT_TIMINGS
} option_id;

typedef struct {
  const char *name;
  option_id id;
  unsigned int commands;
} option_row;

static const option_row option_table[] = {
    {"--input", OPT_INPUT, ALL_COMMANDS},
    {"--output", OPT_OUTPUT, ALL_COMMANDS},
    {"--record-output", OPT_OUTPUT, COMMAND_BIT(SQ_COMMAND_BENCH)},
    {"--words", OPT_WORDS, ENCODE_COMMANDS},
    {"--seed", OPT_SEED, ENCODE_COMMANDS},
    {"--seeds", OPT_SEEDS, COMMAND_BIT(SQ_COMMAND_EXPECT)},
    {"--seed-start", OPT_SEED_START, COMMAND_BIT(SQ_COMMAND_EXPECT)},
    {"--tensor-id", OPT_TENSOR_ID, TENSOR_COMMANDS},
    {"--invocation-id", OPT_INVOCATION_ID, TENSOR_COMMANDS},
    {"--bits", OPT_BITS, TENSOR_COMMANDS},
    {"--scale", OPT_SCALE, ENCODE_COMMANDS},
    {"--backend", OPT_BACKEND, TENSOR_COMMANDS},
    {"--threads", OPT_THREADS, ENCODE_COMMANDS},
    {"--boundary", OPT_BOUNDARY, COMMAND_BIT(SQ_COMMAND_BENCH)},
    {"--transfer-policy", OPT_POLICY, COMMAND_BIT(SQ_COMMAND_BENCH)},
    {"--warmup", OPT_WARMUP, COMMAND_BIT(SQ_COMMAND_BENCH)},
    {"--reps", OPT_REPS, COMMAND_BIT(SQ_COMMAND_BENCH)},
    {"--block-size", OPT_BLOCK_SIZE, ENCODE_COMMANDS},
    {"--grid-size", OPT_GRID_SIZE, ENCODE_COMMANDS},
    {"--k1", OPT_K1, ENCODE_COMMANDS},
    {"--timings", OPT_TIMINGS, ENCODE_COMMANDS},
};

int sq_parse_u64(const char *text, uint64_t *value) {
  uint64_t parsed = 0;
  const char *cursor;

  if (text[0] == '\0') {
    return 0;
  }
  for (cursor = text; *cursor != '\0'; cursor++) {
    unsigned int digit = (unsigned int)(*cursor - '0');

    if (digit > 9 || parsed > (UINT64_MAX - digit) / 10) {
      return 0;
    }
    parsed = parsed * 10 + digit;
  }
  *value = parsed;
  return 1;
}

static int parse_scale(const char *text, float *scale) {
  char *end;
  float parsed;

  errno = 0;
  parsed = strtof(text, &end);
  if (errno == ERANGE || end == text || *end != '\0' || !isfinite(parsed) ||
      parsed < 0.0f) {
    return 0;
  }
  *scale = parsed;
  return 1;
}

static int parse_bounded(const char *text, uint64_t maximum, uint64_t *value) {
  return sq_parse_u64(text, value) && *value != 0 && *value <= maximum;
}

static int parse_k1(const char *text, int *variant) {
  if (strcmp(text, "reference") == 0) {
    *variant = SQ_CUDA_K1_REFERENCE;
  } else if (strcmp(text, "optimized") == 0) {
    *variant = SQ_CUDA_K1_OPTIMIZED;
  } else {
    return 0;
  }
  return 1;
}

const char *sq_k1_name(int variant) {
  return variant == SQ_CUDA_K1_OPTIMIZED ? "optimized" : "reference";
}

static const backend_row *backend_row_for(sq_backend backend) {
  size_t i;

  for (i = 0; i < COUNT_OF(backend_table); i++) {
    if (backend_table[i].backend == backend) {
      return &backend_table[i];
    }
  }
  return NULL;
}

const char *sq_backend_name(sq_backend backend) {
  const backend_row *row = backend_row_for(backend);

  return row != NULL ? row->name : "";
}

const char *sq_transfer_policy_name(sq_transfer_policy policy) {
  size_t i;

  for (i = 0; i < COUNT_OF(policy_table); i++) {
    if (policy_table[i].policy == policy) {
      return policy_table[i].name;
    }
  }
  return "";
}

const sq_legality *sq_cli_legality(sq_backend backend, sq_boundary boundary) {
  size_t i;

  for (i = 0; i < COUNT_OF(legality_table); i++) {
    if (legality_table[i].backend == backend &&
        legality_table[i].boundary == boundary) {
      return &legality_table[i];
    }
  }
  return NULL;
}

static const sq_legality *default_legality(sq_backend backend) {
  size_t i;

  for (i = 0; i < COUNT_OF(legality_table); i++) {
    if (legality_table[i].backend == backend && legality_table[i].is_default) {
      return &legality_table[i];
    }
  }
  return NULL;
}

static int parse_backend(sq_command command, const char *text,
                         sq_backend *backend) {
  size_t i;

  for (i = 0; i < COUNT_OF(backend_table); i++) {
    if ((backend_table[i].commands & COMMAND_BIT(command)) != 0 &&
        strcmp(backend_table[i].name, text) == 0) {
      *backend = backend_table[i].backend;
      return 1;
    }
  }
  return 0;
}

static int parse_boundary(const char *text, sq_boundary *boundary) {
  size_t i;

  for (i = 0; i < COUNT_OF(legality_table); i++) {
    if (legality_table[i].named_on_cli &&
        strcmp(legality_table[i].name, text) == 0) {
      *boundary = legality_table[i].boundary;
      return 1;
    }
  }
  return 0;
}

static int parse_policy(const char *text, sq_transfer_policy *policy) {
  size_t i;

  for (i = 0; i < COUNT_OF(policy_table); i++) {
    if (policy_table[i].named_on_cli &&
        strcmp(policy_table[i].name, text) == 0) {
      *policy = policy_table[i].policy;
      return 1;
    }
  }
  return 0;
}

static const option_row *find_option(sq_command command, const char *name) {
  size_t i;

  for (i = 0; i < COUNT_OF(option_table); i++) {
    if ((option_table[i].commands & COMMAND_BIT(command)) != 0 &&
        strcmp(option_table[i].name, name) == 0) {
      return &option_table[i];
    }
  }
  return NULL;
}

static void options_init(sq_options *options) {
  memset(options, 0, sizeof *options);
  options->backend = SQ_BACKEND_CPU;
  options->boundary = SQ_BOUNDARY_HOST_HOST;
  options->transfer_policy = SQ_TRANSFER_NONE;
  options->seed_start = 1;
  options->bits = SQ_Q8_BITS;
  options->block_size = 256;
  options->warmups = 10;
  options->reps = 30;
  options->k1 = SQ_CUDA_K1_REFERENCE;
}

static sq_status apply_flag(unsigned int flag, int ok, sq_status failure,
                            sq_options *options) {
  if (!ok) {
    return failure;
  }
  options->seen |= flag;
  return SQ_OK;
}

static sq_status apply_option(sq_command command, option_id id,
                              const char *value, sq_options *options) {
  switch (id) {
  case OPT_INPUT:
    options->input_path = value;
    return SQ_OK;
  case OPT_OUTPUT:
    options->output_path = value;
    return SQ_OK;
  case OPT_WORDS:
    options->words_path = value;
    return SQ_OK;
  case OPT_SEED:
    return apply_flag(SQ_SEEN_SEED, sq_parse_u64(value, &options->seed),
                      SQ_ERR_ARGUMENT, options);
  case OPT_SEEDS:
    return apply_flag(SQ_SEEN_SEEDS,
                      sq_parse_u64(value, &options->seeds) &&
                          options->seeds != 0,
                      SQ_ERR_ARGUMENT, options);
  case OPT_SEED_START:
    return sq_parse_u64(value, &options->seed_start) ? SQ_OK : SQ_ERR_ARGUMENT;
  case OPT_TENSOR_ID:
    return sq_parse_u64(value, &options->tensor_id) ? SQ_OK : SQ_ERR_ARGUMENT;
  case OPT_INVOCATION_ID:
    return sq_parse_u64(value, &options->invocation_id) ? SQ_OK
                                                        : SQ_ERR_ARGUMENT;
  case OPT_BITS:
    return sq_parse_u64(value, &options->bits) ? SQ_OK : SQ_ERR_ARGUMENT;
  case OPT_SCALE:
    return apply_flag(SQ_SEEN_SCALE, parse_scale(value, &options->scale),
                      SQ_ERR_SCALE, options);
  case OPT_BACKEND:
    return parse_backend(command, value, &options->backend) ? SQ_OK
                                                            : SQ_ERR_ARGUMENT;
  case OPT_THREADS:
    return apply_flag(
        SQ_SEEN_THREADS,
        parse_bounded(value, SQ_MAX_CPU_THREADS, &options->threads),
        SQ_ERR_ARGUMENT, options);
  case OPT_BOUNDARY:
    return apply_flag(SQ_SEEN_BOUNDARY,
                      parse_boundary(value, &options->boundary),
                      SQ_ERR_ARGUMENT, options);
  case OPT_POLICY:
    return apply_flag(SQ_SEEN_POLICY,
                      parse_policy(value, &options->transfer_policy),
                      SQ_ERR_ARGUMENT, options);
  case OPT_WARMUP:
    return sq_parse_u64(value, &options->warmups) ? SQ_OK : SQ_ERR_ARGUMENT;
  case OPT_REPS:
    return sq_parse_u64(value, &options->reps) ? SQ_OK : SQ_ERR_ARGUMENT;
  case OPT_BLOCK_SIZE:
    return apply_flag(
        SQ_SEEN_BLOCK_SIZE,
        parse_bounded(value, SQ_CUDA_MAX_BLOCK_SIZE, &options->block_size),
        SQ_ERR_ARGUMENT, options);
  case OPT_GRID_SIZE:
    return apply_flag(
        SQ_SEEN_GRID_SIZE,
        parse_bounded(value, SQ_CUDA_MAX_GRID_SIZE, &options->grid_size),
        SQ_ERR_ARGUMENT, options);
  case OPT_K1:
    return apply_flag(SQ_SEEN_K1, parse_k1(value, &options->k1),
                      SQ_ERR_ARGUMENT, options);
  case OPT_TIMINGS:
    break;
  }
  return SQ_ERR_ARGUMENT;
}

sq_status sq_cli_parse(sq_command command, int argc, char **argv,
                       sq_options *options) {
  int i;

  options_init(options);
  for (i = 2; i < argc; i++) {
    const option_row *row = find_option(command, argv[i]);
    sq_status status;

    if (row != NULL && row->id == OPT_TIMINGS) {
      options->seen |= SQ_SEEN_TIMINGS;
      continue;
    }
    if (row == NULL || i + 1 >= argc) {
      return SQ_ERR_ARGUMENT;
    }
    status = apply_option(command, row->id, argv[i + 1], options);
    if (status != SQ_OK) {
      return status;
    }
    i++;
  }
  return SQ_OK;
}

/* Boundary and transfer policy for a bench line, decided by the legality
   table. */
static sq_status resolve_bench(sq_options *options,
                               const sq_legality **resolved) {
  const sq_legality *row;
  int policy_seen = (options->seen & SQ_SEEN_POLICY) != 0;

  if ((options->seen & SQ_SEEN_BOUNDARY) != 0) {
    row = sq_cli_legality(options->backend, options->boundary);
  } else {
    row = default_legality(options->backend);
  }
  if (row == NULL || (policy_seen && !row->transfers)) {
    return SQ_ERR_ARGUMENT;
  }
  options->boundary = row->boundary;
  if (!row->transfers) {
    options->transfer_policy = SQ_TRANSFER_NONE;
  } else if (!policy_seen) {
    options->transfer_policy = SQ_TRANSFER_PAGEABLE;
  }
  *resolved = row;
  return SQ_OK;
}

static sq_status check_required(sq_command command, const sq_options *options) {
  int seed_seen = (options->seen & SQ_SEEN_SEED) != 0;

  switch (command) {
  case SQ_COMMAND_DECOMPRESS:
    break;
  case SQ_COMMAND_COMPRESS:
    if (!seed_seen) {
      return SQ_ERR_ARGUMENT;
    }
    break;
  case SQ_COMMAND_BENCH:
    if (!seed_seen || options->reps == 0) {
      return SQ_ERR_ARGUMENT;
    }
    break;
  case SQ_COMMAND_EXPECT:
    if ((options->seen & SQ_SEEN_SEEDS) == 0 ||
        options->seed_start > UINT64_MAX - (options->seeds - 1)) {
      return SQ_ERR_ARGUMENT;
    }
    break;
  }
  if (options->input_path == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  if (command != SQ_COMMAND_BENCH && options->output_path == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  return SQ_OK;
}

sq_status sq_cli_validate(sq_command command, sq_options *options,
                          int cuda_available) {
  const backend_row *backend = backend_row_for(options->backend);
  const unsigned int cuda_only =
      SQ_SEEN_BLOCK_SIZE | SQ_SEEN_GRID_SIZE | SQ_SEEN_K1;
  int is_cuda = backend != NULL && backend->is_cuda;
  int needs_cuda = is_cuda;
  sq_status status = check_required(command, options);

  if (status != SQ_OK || command == SQ_COMMAND_DECOMPRESS) {
    return status;
  }
  if (options->bits != SQ_Q4_BITS && options->bits != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  if (!is_cuda && (options->seen & cuda_only) != 0) {
    return SQ_ERR_ARGUMENT;
  }
  if (command == SQ_COMMAND_BENCH) {
    const sq_legality *row = NULL;

    status = resolve_bench(options, &row);
    if (status != SQ_OK) {
      return status;
    }
    needs_cuda = needs_cuda || row->needs_cuda;
  }
  /* Only CUDA has stage timings. A CUDA bench always reports its stage columns,
     so there --timings is already satisfied; compress prints them on request.
   */
  if ((options->seen & SQ_SEEN_TIMINGS) != 0 && !is_cuda) {
    return SQ_ERR_TIMINGS_BACKEND;
  }
  if (needs_cuda && !cuda_available) {
    return SQ_ERR_CUDA_UNAVAILABLE;
  }
  return SQ_OK;
}

#ifdef _WIN32
typedef __int64 file_offset;
#define FILE_SEEK _fseeki64
#define FILE_TELL _ftelli64
#else
typedef off_t file_offset;
#define FILE_SEEK fseeko
#define FILE_TELL ftello
#endif

int sq_read_file(const char *path, uint8_t **bytes, size_t *size) {
  FILE *file;
  file_offset length;
  uint8_t *buffer = NULL;
  size_t done = 0;

  *bytes = NULL;
  *size = 0;
  file = fopen(path, "rb");
  if (file == NULL) {
    return 0;
  }
  if (FILE_SEEK(file, 0, SEEK_END) != 0) {
    goto fail;
  }
  length = FILE_TELL(file);
  if (length < 0 || (uintmax_t)length > (uintmax_t)SIZE_MAX) {
    goto fail;
  }
  if (FILE_SEEK(file, 0, SEEK_SET) != 0) {
    goto fail;
  }
  if (length != 0) {
    buffer = (uint8_t *)malloc((size_t)length);
    if (buffer == NULL) {
      goto fail;
    }
    while (done < (size_t)length) {
      size_t want = (size_t)length - done;

      if (want > SQ_READ_CHUNK) {
        want = SQ_READ_CHUNK;
      }
      if (fread(buffer + done, 1, want, file) != want) {
        goto fail;
      }
      done += want;
    }
  }
  if (fclose(file) != 0) {
    free(buffer);
    return 0;
  }
  *bytes = buffer;
  *size = (size_t)length;
  return 1;

fail:
  free(buffer);
  (void)fclose(file);
  return 0;
}

int sq_write_file(const char *path, const uint8_t *bytes, size_t size) {
  FILE *file = fopen(path, "wb");
  int ok;

  if (file == NULL) {
    return 0;
  }
  ok = size == 0 || fwrite(bytes, 1, size, file) == size;
  if (fclose(file) != 0) {
    ok = 0;
  }
  return ok;
}
