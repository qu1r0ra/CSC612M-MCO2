#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "byteorder.h"
#include "codec.h"
#include "quantizer.h"
#include "quantizer_avx2.h"
#include "quantizer_cuda.h"
#include "rng_cpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif

#define SQ_MAX_CPU_THREADS 256

static void usage(FILE *stream) {
  (void)fprintf(
      stream,
      "Usage:\n"
      "  stoquant compress --input INPUT.f32 --output RECORD --seed UINT64\n"
      "      [--backend cpu|cpu-avx2|cuda] [--bits 4|8] [--tensor-id UINT32]\n"
      "      [--invocation-id UINT32] [--scale FP32] [--words WORDS.u32]\n"
      "      [--threads UINT32] [--block-size UINT32] [--grid-size UINT32]\n"
      "      [--k1 reference|optimized] [--timings]\n"
      "  stoquant bench --input INPUT.f32 --seed UINT64\n"
      "      [--output RECORD | --record-output RECORD]\n"
      "      [--backend cpu|cpu-avx2|cuda] [--bits 4|8] [--tensor-id UINT32]\n"
      "      [--invocation-id UINT32] [--scale FP32] [--words WORDS.u32]\n"
      "      [--threads UINT32]\n"
      "      [--boundary resident|resident-graph|host-origin|gpu-origin]\n"
      "      [--transfer-policy pageable|pinned] [--warmup UINT32 (default "
      "10)]\n"
      "      [--reps UINT32 (default 30)]\n"
      "      [--block-size UINT32] [--grid-size UINT32]\n"
      "      [--k1 reference|optimized] [--timings]\n"
      "  stoquant decompress --input RECORD --output OUTPUT.f32\n"
      "  stoquant expect --input INPUT.f32 --output SUMS.f64 --seeds UINT64\n"
      "      [--seed-start UINT64 (default 1)] [--backend cpu|cuda]\n"
      "      [--bits 4|8] [--tensor-id UINT32] [--invocation-id UINT32]\n"
      "\n"
      "Input and output tensors use little-endian raw FP32. Prescribed\n"
      "random words use little-endian raw uint32, one per element.\n"
      "expect writes little-endian FP64 per-element sums of the decoded\n"
      "values, then per-element sums of their squares.\n"
      "cpu-avx2 writes the same bytes as cpu; --threads defaults to the\n"
      "processor count in the process affinity mask (Windows) or online "
      "(elsewhere).\n"
      "--k1 (cuda only) picks the scale-stage kernels; both write the same "
      "bytes\n"
      "and reference is the default.\n");
}

static int read_file(const char *path, uint8_t **bytes, size_t *size) {
  FILE *file;
  long length;
  uint8_t *buffer = NULL;
  size_t read_count;

  *bytes = NULL;
  *size = 0;
  file = fopen(path, "rb");
  if (file == NULL) {
    return 0;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    goto fail;
  }
  length = ftell(file);
  if (length < 0 || (uintmax_t)length > (uintmax_t)SIZE_MAX) {
    goto fail;
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    goto fail;
  }
  if (length != 0) {
    buffer = (uint8_t *)malloc((size_t)length);
    if (buffer == NULL) {
      goto fail;
    }
    read_count = fread(buffer, 1, (size_t)length, file);
    if (read_count != (size_t)length) {
      goto fail;
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

static int write_file(const char *path, const uint8_t *bytes, size_t size) {
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

static int parse_u64(const char *text, uint64_t *value) {
  char *end;
  unsigned long long parsed;

  if (text[0] == '\0' || text[0] == '-') {
    return 0;
  }
  errno = 0;
  parsed = strtoull(text, &end, 0);
  if (errno == ERANGE || end == text || *end != '\0') {
    return 0;
  }
  *value = (uint64_t)parsed;
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

/* Checked here, outside the /arch:AVX2 translation unit: CPU support plus
 * OS-saved YMM state. */
static int cpu_has_avx2(void) {
#if defined(_MSC_VER)
  int info[4];

  __cpuid(info, 0);
  if (info[0] < 7) {
    return 0;
  }
  __cpuid(info, 1);
  if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0) {
    return 0;
  }
  if ((_xgetbv(0) & 6) != 6) {
    return 0;
  }
  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#else
  return 0;
#endif
}

static int default_thread_count(void) {
#ifdef _WIN32
  DWORD_PTR process_mask, system_mask;
  long count = 0;

  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask,
                             &system_mask)) {
    for (; process_mask != 0; process_mask &= process_mask - 1) {
      count++;
    }
  }
#else
  long count = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (count < 1) {
    return 1;
  }
  return count > SQ_MAX_CPU_THREADS ? SQ_MAX_CPU_THREADS : (int)count;
}

static int parse_threads(const char *text, uint64_t *threads) {
  return parse_u64(text, threads) && *threads != 0 &&
         *threads <= SQ_MAX_CPU_THREADS;
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

static const char *k1_name(int variant) {
  return variant == SQ_CUDA_K1_OPTIMIZED ? "optimized" : "reference";
}

/* Sets *threads to 0 for the scalar comparator, else to the requested AVX2 team
 * size. */
static sq_status resolve_cpu_threads(const char *backend, int threads_seen,
                                     uint64_t requested, int *threads) {
  *threads = 0;
  if (strcmp(backend, "cpu-avx2") != 0) {
    return threads_seen ? SQ_ERR_ARGUMENT : SQ_OK;
  }
  if (!cpu_has_avx2()) {
    return SQ_ERR_AVX2_UNAVAILABLE;
  }
  *threads = threads_seen ? (int)requested : default_thread_count();
  return SQ_OK;
}

static int is_backend(const char *name) {
  return strcmp(name, "cpu") == 0 || strcmp(name, "cpu-avx2") == 0 ||
         strcmp(name, "cuda") == 0;
}

static int is_cpu_backend(const char *name) {
  return strcmp(name, "cpu") == 0 || strcmp(name, "cpu-avx2") == 0;
}

/* One CPU compression; threads 0 runs the scalar comparator, otherwise the AVX2
 * one. */
static sq_status cpu_compute_scale(const float *values, size_t count,
                                   float *scale, float *partials,
                                   size_t partial_capacity, int threads) {
  if (threads == 0) {
    return sq_compute_scale_with_workspace(values, count, scale, partials,
                                           partial_capacity);
  }
  return sq_avx2_compute_scale_with_workspace(values, count, scale, partials,
                                              partial_capacity, threads);
}

static void cpu_rng_words(const sq_rng_stream *stream, size_t count,
                          uint32_t *words, int threads) {
  if (threads == 0) {
    sq_rng_words_cpu(stream, (uint64_t)count, words);
  } else {
    sq_avx2_rng_words(stream, (uint64_t)count, words, threads);
  }
}

static sq_status cpu_encode_payload(uint8_t bits, const float *values,
                                    size_t count, float scale,
                                    const uint32_t *words, uint8_t *payload,
                                    int threads) {
  if (threads == 0) {
    return sq_encode_payload(bits, values, count, scale, words, payload);
  }
  return sq_avx2_encode_payload(bits, values, count, scale, words, payload,
                                threads);
}

static sq_status compress_file(int argc, char **argv) {
  const char *input_path = NULL, *output_path = NULL, *words_path = NULL;
  const char *backend = "cpu";
  uint64_t seed = 0, tensor_id = 0, invocation_id = 0, bits = SQ_Q8_BITS;
  uint64_t block_size = 256, grid_size = 0, requested_threads = 0;
  float prescribed_scale = 0.0f, scale;
  int seed_seen = 0, scale_seen = 0, timings_seen = 0, threads_seen = 0;
  int block_size_seen = 0, grid_size_seen = 0, threads, i;
  int k1 = SQ_CUDA_K1_REFERENCE, k1_seen = 0;
  uint8_t *input_bytes = NULL, *word_bytes = NULL, *record = NULL;
  float *values = NULL, *scale_partials = NULL;
  uint32_t *words = NULL;
  uint8_t *codes;
  size_t input_size = 0, word_size = 0, count, i_size;
  sq_rng_stream stream;
  sq_status status;

  for (i = 2; i < argc; i++) {
    const char *option = argv[i];
    const char *value;
    if (strcmp(option, "--timings") == 0) {
      timings_seen = 1;
      continue;
    }
    if (i + 1 >= argc) {
      return SQ_ERR_ARGUMENT;
    }
    value = argv[++i];
    if (strcmp(option, "--input") == 0) {
      input_path = value;
    } else if (strcmp(option, "--output") == 0) {
      output_path = value;
    } else if (strcmp(option, "--words") == 0) {
      words_path = value;
    } else if (strcmp(option, "--seed") == 0) {
      seed_seen = parse_u64(value, &seed);
      if (!seed_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--tensor-id") == 0) {
      if (!parse_u64(value, &tensor_id)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--invocation-id") == 0) {
      if (!parse_u64(value, &invocation_id)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--bits") == 0) {
      if (!parse_u64(value, &bits)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--scale") == 0) {
      scale_seen = parse_scale(value, &prescribed_scale);
      if (!scale_seen) {
        return SQ_ERR_SCALE;
      }
    } else if (strcmp(option, "--backend") == 0) {
      if (!is_backend(value)) {
        return SQ_ERR_ARGUMENT;
      }
      backend = value;
    } else if (strcmp(option, "--threads") == 0) {
      threads_seen = parse_threads(value, &requested_threads);
      if (!threads_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--block-size") == 0) {
      if (!parse_u64(value, &block_size) || block_size == 0 ||
          block_size > 1024) {
        return SQ_ERR_ARGUMENT;
      }
      block_size_seen = 1;
    } else if (strcmp(option, "--grid-size") == 0) {
      if (!parse_u64(value, &grid_size) || grid_size == 0 ||
          grid_size > 65535) {
        return SQ_ERR_ARGUMENT;
      }
      grid_size_seen = 1;
    } else if (strcmp(option, "--k1") == 0) {
      k1_seen = parse_k1(value, &k1);
      if (!k1_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else {
      return SQ_ERR_ARGUMENT;
    }
  }
  if (input_path == NULL || output_path == NULL || !seed_seen) {
    return SQ_ERR_ARGUMENT;
  }
  if (bits != SQ_Q4_BITS && bits != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  if ((block_size_seen || grid_size_seen || k1_seen) &&
      strcmp(backend, "cuda") != 0) {
    return SQ_ERR_ARGUMENT;
  }
  if (timings_seen && strcmp(backend, "cuda") != 0) {
    return SQ_ERR_TIMINGS_BACKEND;
  }
#ifndef SQ_ENABLE_CUDA
  if (strcmp(backend, "cuda") == 0) {
    return SQ_ERR_CUDA_UNAVAILABLE;
  }
#endif
  status =
      resolve_cpu_threads(backend, threads_seen, requested_threads, &threads);
  if (status != SQ_OK) {
    return status;
  }
  if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX) {
    return SQ_ERR_ID_OVERFLOW;
  }

  if (!read_file(input_path, &input_bytes, &input_size)) {
    return SQ_ERR_IO;
  }
  if (input_size % sizeof(uint32_t) != 0) {
    status = SQ_ERR_PAYLOAD_LENGTH;
    goto done;
  }
  count = input_size / sizeof(uint32_t);
  if (count > SIZE_MAX / sizeof *values || count > SIZE_MAX - SQ_HEADER_SIZE) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  values = count == 0 ? NULL : (float *)malloc(count * sizeof *values);
  if (count != 0 && values == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  for (i_size = 0; i_size < count; i_size++) {
    uint32_t bits32 = sq_load_u32_le(input_bytes + 4 * i_size);
    memcpy(&values[i_size], &bits32, sizeof bits32);
  }

  if (strcmp(backend, "cuda") == 0) {
#ifdef SQ_ENABLE_CUDA
    sq_cuda_timings cuda_timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    size_t payload_size = (bits == SQ_Q4_BITS) ? (count + 1) / 2 : count;

    if (words_path != NULL) {
      if (!read_file(words_path, &word_bytes, &word_size)) {
        status = SQ_ERR_IO;
        goto done;
      }
      if (count > SIZE_MAX / sizeof(uint32_t) ||
          word_size != count * sizeof(uint32_t)) {
        status = SQ_ERR_PAYLOAD_LENGTH;
        goto done;
      }
      words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
      if (count != 0 && words == NULL) {
        status = SQ_ERR_MEMORY;
        goto done;
      }
      for (i_size = 0; i_size < count; i_size++) {
        words[i_size] = sq_load_u32_le(word_bytes + 4 * i_size);
      }
    }

    record = (uint8_t *)malloc(SQ_HEADER_SIZE + payload_size);
    if (record == NULL) {
      status = SQ_ERR_MEMORY;
      goto done;
    }
    if (sq_cuda_select_k1(k1) != 0) {
      status = SQ_ERR_ARGUMENT;
      goto done;
    }
    status = sq_cuda_compress(
        (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
        scale_seen, prescribed_scale, words, (int)block_size, (int)grid_size,
        record + SQ_HEADER_SIZE, &scale, timings_seen, &cuda_timings);
    if (status != SQ_OK) {
      goto done;
    }
    status = sq_header_encode((uint8_t)bits, (uint64_t)count, scale, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (!write_file(output_path, record, SQ_HEADER_SIZE + payload_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (timings_seen) {
      (void)fprintf(stderr,
                    "{\"k1_ms\":%.6f,\"k2_ms\":%.6f,\"k3_ms\":%.6f,"
                    "\"h2d_ms\":%.6f,\"d2h_ms\":%.6f}\n",
                    cuda_timings.k1_ms, cuda_timings.k2_ms, cuda_timings.k3_ms,
                    cuda_timings.h2d_ms, cuda_timings.d2h_ms);
    }
    status = SQ_OK;
    goto done;
#endif
  }

  if (scale_seen) {
    scale = prescribed_scale;
  } else if (threads == 0) {
    status = sq_compute_scale(values, count, &scale);
    if (status != SQ_OK) {
      goto done;
    }
  } else {
    size_t partial_count = sq_scale_workspace_elements(count);

    if (partial_count == SIZE_MAX ||
        partial_count > SIZE_MAX / sizeof *scale_partials) {
      status = SQ_ERR_MEMORY;
      goto done;
    }
    if (partial_count != 0) {
      scale_partials = (float *)malloc(partial_count * sizeof *scale_partials);
      if (scale_partials == NULL) {
        status = SQ_ERR_MEMORY;
        goto done;
      }
    }
    status = cpu_compute_scale(values, count, &scale, scale_partials,
                               partial_count, threads);
    if (status != SQ_OK) {
      goto done;
    }
  }

  if (words_path != NULL) {
    if (!read_file(words_path, &word_bytes, &word_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (count > SIZE_MAX / sizeof(uint32_t) ||
        word_size != count * sizeof(uint32_t)) {
      status = SQ_ERR_PAYLOAD_LENGTH;
      goto done;
    }
  }

  words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
  if (count != 0 && words == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  if (words_path != NULL) {
    for (i_size = 0; i_size < count; i_size++) {
      words[i_size] = sq_load_u32_le(word_bytes + 4 * i_size);
    }
  } else {
    if (sq_rng_stream_init(&stream, seed, tensor_id, invocation_id) !=
        SQ_RNG_OK) {
      status = SQ_ERR_ID_OVERFLOW;
      goto done;
    }
    cpu_rng_words(&stream, count, words, threads);
  }

  {
    size_t payload_size = (bits == SQ_Q4_BITS) ? (count + 1) / 2 : count;
    record = (uint8_t *)malloc(SQ_HEADER_SIZE + payload_size);
    if (record == NULL) {
      status = SQ_ERR_MEMORY;
      goto done;
    }
    status = sq_header_encode((uint8_t)bits, (uint64_t)count, scale, record);
    if (status != SQ_OK) {
      goto done;
    }
    codes = record + SQ_HEADER_SIZE;
    status = cpu_encode_payload((uint8_t)bits, values, count, scale, words,
                                codes, threads);
    if (status != SQ_OK) {
      goto done;
    }
    if (!write_file(output_path, record, SQ_HEADER_SIZE + payload_size)) {
      status = SQ_ERR_IO;
    }
  }

done:
  free(scale_partials);
  free(record);
  free(words);
  free(word_bytes);
  free(values);
  free(input_bytes);
  return status;
}

#ifdef _WIN32
typedef LARGE_INTEGER bench_clock_frequency;
static int bench_clock_init(bench_clock_frequency *frequency) {
  return QueryPerformanceFrequency(frequency) != 0;
}
static int bench_clock_now_ms(const bench_clock_frequency *frequency,
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
typedef int bench_clock_frequency;
static int bench_clock_init(bench_clock_frequency *frequency) {
  (void)frequency;
  return 1;
}
static int bench_clock_now_ms(const bench_clock_frequency *frequency,
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

static sq_status bench_cpu_compress_one(
    uint8_t bit_width, const float *values, size_t count, uint64_t seed,
    uint64_t tensor_id, uint64_t invocation_id, int prescribed_scale_seen,
    float prescribed_scale, const uint32_t *prescribed_words,
    uint32_t *generated_words, float *scale_partials,
    size_t scale_partial_capacity, int threads, uint8_t *record) {
  float scale;
  sq_rng_stream stream;
  sq_status status;
  if (prescribed_scale_seen) {
    scale = prescribed_scale;
  } else {
    status = cpu_compute_scale(values, count, &scale, scale_partials,
                               scale_partial_capacity, threads);
    if (status != SQ_OK) {
      return status;
    }
  }
  if (prescribed_words == NULL) {
    status = sq_rng_stream_init(&stream, seed, tensor_id, invocation_id);
    if (status != SQ_OK) {
      return SQ_ERR_ID_OVERFLOW;
    }
    cpu_rng_words(&stream, count, generated_words, threads);
  }
  status = sq_header_encode(bit_width, (uint64_t)count, scale, record);
  if (status != SQ_OK) {
    return status;
  }
  return cpu_encode_payload(bit_width, values, count, scale,
                            prescribed_words != NULL ? prescribed_words
                                                     : generated_words,
                            record + SQ_HEADER_SIZE, threads);
}

typedef enum {
  BENCH_SAMPLE_WALL_TIME,
  BENCH_SAMPLE_K1_TIME,
  BENCH_SAMPLE_K2_TIME,
  BENCH_SAMPLE_K3_TIME,
  BENCH_SAMPLE_H2D_TIME,
  BENCH_SAMPLE_D2H_TIME,
  BENCH_SAMPLE_CPU_TIME
} sq_bench_timing_field;

static void print_double_array(const sq_bench_sample *samples, uint64_t reps,
                               sq_bench_timing_field field) {
  uint64_t i;
  putchar('[');
  for (i = 0; i < reps; i++) {
    double value;
    switch (field) {
    case BENCH_SAMPLE_WALL_TIME:
      value = samples[i].wall_ms;
      break;
    case BENCH_SAMPLE_K1_TIME:
      value = samples[i].k1_ms;
      break;
    case BENCH_SAMPLE_K2_TIME:
      value = samples[i].k2_ms;
      break;
    case BENCH_SAMPLE_K3_TIME:
      value = samples[i].k3_ms;
      break;
    case BENCH_SAMPLE_H2D_TIME:
      value = samples[i].h2d_ms;
      break;
    case BENCH_SAMPLE_D2H_TIME:
      value = samples[i].d2h_ms;
      break;
    case BENCH_SAMPLE_CPU_TIME:
      value = samples[i].cpu_ms;
      break;
    default:
      value = 0.0;
      break;
    }
    if (i != 0) {
      putchar(',');
    }
    printf("%.9f", value);
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

static void
print_bench_json(const char *backend, const char *boundary,
                 const char *transfer_policy, uint8_t bit_width, size_t count,
                 uint64_t seed, uint64_t tensor_id, uint64_t invocation_id,
                 uint64_t warmups, uint64_t reps, int prescribed_scale_seen,
                 float prescribed_scale, int block_size, int grid_size,
                 const char *k1, int team_size, size_t payload_bytes,
                 const sq_bench_sample *samples, double capture_ms) {
  const uint64_t id_step = strcmp(boundary, "resident-graph") == 0 ? 0 : 1;

  printf("{\"configuration\":{\"backend\":\"%s\",\"bits\":%u,"
         "\"count\":%llu,\"seed\":%llu,\"tensor_id\":%llu,"
         "\"invocation_id\":%llu,\"warmup\":%llu,\"reps\":%llu,"
         "\"repetition_invocation_ids\":",
         backend, (unsigned int)bit_width, (unsigned long long)count,
         (unsigned long long)seed, (unsigned long long)tensor_id,
         (unsigned long long)invocation_id, (unsigned long long)warmups,
         (unsigned long long)reps);
  print_invocation_ids(invocation_id, reps, 0, id_step);
  printf(",\"warmup_invocation_ids\":");
  print_invocation_ids(invocation_id, warmups, reps, id_step);
  printf(",\"boundary\":\"%s\",\"transfer_policy\":\"%s\","
         "\"block_size\":%d,\"grid_size\":%d,",
         boundary, transfer_policy, block_size, grid_size);
  if (k1 != NULL) {
    printf("\"k1\":\"%s\",", k1);
  }
  /* The team size OpenMP grants a probe region; with dynamic teams off the
     timed regions get the same size. */
  if (team_size != 0) {
    printf("\"threads\":%d,", team_size);
  }
  printf("\"prescribed_scale\":");
  if (prescribed_scale_seen) {
    printf("%.9g", (double)prescribed_scale);
  } else {
    printf("null");
  }
  printf("},\"samples_ms\":");
  print_double_array(samples, reps, BENCH_SAMPLE_WALL_TIME);
  if (strcmp(backend, "cuda") == 0) {
    if (strcmp(boundary, "resident-graph") != 0) {
      printf(",\"k1_ms\":");
      print_double_array(samples, reps, BENCH_SAMPLE_K1_TIME);
      printf(",\"k2_ms\":");
      print_double_array(samples, reps, BENCH_SAMPLE_K2_TIME);
      printf(",\"k3_ms\":");
      print_double_array(samples, reps, BENCH_SAMPLE_K3_TIME);
      if (strcmp(boundary, "host-origin") == 0) {
        printf(",\"h2d_ms\":");
        print_double_array(samples, reps, BENCH_SAMPLE_H2D_TIME);
      }
      if (strcmp(boundary, "host-origin") == 0 ||
          strcmp(boundary, "gpu-origin") == 0) {
        printf(",\"d2h_ms\":");
        print_double_array(samples, reps, BENCH_SAMPLE_D2H_TIME);
      }
    } else {
      printf(",\"capture_and_instantiate_ms\":%.6f", capture_ms);
    }
  } else if (strcmp(boundary, "gpu-origin") == 0) {
    printf(",\"d2h_ms\":");
    print_double_array(samples, reps, BENCH_SAMPLE_D2H_TIME);
    printf(",\"cpu_ms\":");
    print_double_array(samples, reps, BENCH_SAMPLE_CPU_TIME);
  }
  printf(",\"header_bytes\":%d,\"payload_bytes\":%llu}\n", SQ_HEADER_SIZE,
         (unsigned long long)payload_bytes);
}

#ifdef SQ_ENABLE_CUDA
/*
 * GPU-origin CPU path: the input starts on the device. Each run times one full
 * D2H into the landing buffer (d2h_ms), then CPU compression from it (cpu_ms).
 */
static sq_status bench_cpu_gpu_origin(
    uint8_t bits, const float *values, size_t count, uint64_t seed,
    uint64_t tensor_id, uint64_t invocation_id, int scale_seen,
    float prescribed_scale, const uint32_t *prescribed_words,
    uint32_t *generated_words, float *scale_partials,
    size_t scale_partial_count, sq_cuda_transfer_policy transfer_policy,
    uint64_t warmups, uint64_t reps, uint8_t *record, const char *output_path,
    size_t record_size, sq_bench_sample *samples) {
  sq_cuda_staging *staging = NULL;
  bench_clock_frequency clock_frequency;
  const float *landing;
  double d2h_ms;
  sq_status status;

  status = sq_cuda_staging_create(values, count, transfer_policy, &staging);
  if (status != SQ_OK) {
    return status;
  }
  status = sq_cuda_staging_download(staging, &landing, &d2h_ms);
  if (status == SQ_OK) {
    status = bench_cpu_compress_one(
        bits, landing, count, seed, tensor_id, invocation_id, scale_seen,
        prescribed_scale, prescribed_words, generated_words, scale_partials,
        scale_partial_count, 0, record);
  }
  if (status != SQ_OK) {
    goto done;
  }
  if (output_path != NULL && !write_file(output_path, record, record_size)) {
    status = SQ_ERR_IO;
    goto done;
  }
  if (!bench_clock_init(&clock_frequency)) {
    status = SQ_ERR_CLOCK;
    goto done;
  }
  for (uint64_t run = 0; run < warmups + reps; run++) {
    double start_ms, cpu_start_ms, stop_ms;
    uint64_t run_offset = run < warmups ? reps + run : run - warmups;
    if (!bench_clock_now_ms(&clock_frequency, &start_ms)) {
      status = SQ_ERR_CLOCK;
      goto done;
    }
    status = sq_cuda_staging_download(staging, &landing, &d2h_ms);
    if (status != SQ_OK) {
      goto done;
    }
    if (!bench_clock_now_ms(&clock_frequency, &cpu_start_ms)) {
      status = SQ_ERR_CLOCK;
      goto done;
    }
    status = bench_cpu_compress_one(
        bits, landing, count, seed, tensor_id, invocation_id + run_offset,
        scale_seen, prescribed_scale, prescribed_words, generated_words,
        scale_partials, scale_partial_count, 0, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (!bench_clock_now_ms(&clock_frequency, &stop_ms)) {
      status = SQ_ERR_CLOCK;
      goto done;
    }
    if (run >= warmups) {
      sq_bench_sample *sample = &samples[run - warmups];
      sample->wall_ms = stop_ms - start_ms;
      sample->d2h_ms = d2h_ms;
      sample->cpu_ms = stop_ms - cpu_start_ms;
    }
  }
  status = SQ_OK;

done:
  sq_cuda_staging_destroy(staging);
  return status;
}
#endif

static sq_status bench_file(int argc, char **argv) {
  const char *input_path = NULL, *output_path = NULL, *words_path = NULL;
  const char *backend = "cpu", *boundary_name = "host-origin";
  const char *transfer_policy_name = "pageable";
  double capture_ms = 0.0;
  uint64_t seed = 0, tensor_id = 0, invocation_id = 0;
  uint64_t bits = SQ_Q8_BITS, block_size = 256, grid_size = 0;
  uint64_t warmups = 10, reps = 30, total_runs, requested_threads = 0;
  float prescribed_scale = 0.0f;
#ifdef SQ_ENABLE_CUDA
  float scale;
#endif
  int seed_seen = 0, scale_seen = 0, boundary_seen = 0, policy_seen = 0;
  int threads_seen = 0, threads, team_size = 0, cpu_gpu_origin;
  int transfers;
  int block_size_seen = 0, grid_size_seen = 0, i;
  int k1 = SQ_CUDA_K1_REFERENCE, k1_seen = 0;
  uint8_t *input_bytes = NULL, *word_bytes = NULL, *record = NULL;
  float *values = NULL, *scale_partials = NULL;
  uint32_t *words = NULL, *generated_words = NULL;
  sq_bench_sample *samples = NULL;
  size_t input_size = 0, word_size = 0, count = 0, payload_bytes;
  size_t record_size, scale_partial_count;
  bench_clock_frequency clock_frequency;
  sq_status status = SQ_ERR_ARGUMENT;

  for (i = 2; i < argc; i++) {
    const char *option = argv[i];
    const char *value;

    if (strcmp(option, "--timings") == 0) {
      continue;
    }
    if (i + 1 >= argc) {
      return SQ_ERR_ARGUMENT;
    }
    value = argv[++i];
    if (strcmp(option, "--input") == 0) {
      input_path = value;
    } else if (strcmp(option, "--output") == 0 ||
               strcmp(option, "--record-output") == 0) {
      output_path = value;
    } else if (strcmp(option, "--words") == 0) {
      words_path = value;
    } else if (strcmp(option, "--seed") == 0) {
      seed_seen = parse_u64(value, &seed);
      if (!seed_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--tensor-id") == 0) {
      if (!parse_u64(value, &tensor_id)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--invocation-id") == 0) {
      if (!parse_u64(value, &invocation_id)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--bits") == 0) {
      if (!parse_u64(value, &bits)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--scale") == 0) {
      scale_seen = parse_scale(value, &prescribed_scale);
      if (!scale_seen) {
        return SQ_ERR_SCALE;
      }
    } else if (strcmp(option, "--backend") == 0) {
      if (!is_backend(value)) {
        return SQ_ERR_ARGUMENT;
      }
      backend = value;
    } else if (strcmp(option, "--threads") == 0) {
      threads_seen = parse_threads(value, &requested_threads);
      if (!threads_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--boundary") == 0) {
      if (strcmp(value, "resident") != 0 &&
          strcmp(value, "resident-graph") != 0 &&
          strcmp(value, "host-origin") != 0 &&
          strcmp(value, "gpu-origin") != 0) {
        return SQ_ERR_ARGUMENT;
      }
      boundary_name = value;
      boundary_seen = 1;
    } else if (strcmp(option, "--transfer-policy") == 0) {
      if (strcmp(value, "pageable") != 0 && strcmp(value, "pinned") != 0) {
        return SQ_ERR_ARGUMENT;
      }
      transfer_policy_name = value;
      policy_seen = 1;
    } else if (strcmp(option, "--warmup") == 0) {
      if (!parse_u64(value, &warmups)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--reps") == 0) {
      if (!parse_u64(value, &reps)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--block-size") == 0) {
      if (!parse_u64(value, &block_size) || block_size == 0 ||
          block_size > SQ_CUDA_MAX_BLOCK_SIZE) {
        return SQ_ERR_ARGUMENT;
      }
      block_size_seen = 1;
    } else if (strcmp(option, "--grid-size") == 0) {
      if (!parse_u64(value, &grid_size) || grid_size == 0 ||
          grid_size > SQ_CUDA_MAX_GRID_SIZE) {
        return SQ_ERR_ARGUMENT;
      }
      grid_size_seen = 1;
    } else if (strcmp(option, "--k1") == 0) {
      k1_seen = parse_k1(value, &k1);
      if (!k1_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else {
      return SQ_ERR_ARGUMENT;
    }
  }

  if (input_path == NULL || !seed_seen || reps == 0) {
    return SQ_ERR_ARGUMENT;
  }
  if (bits != SQ_Q4_BITS && bits != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
  /*
   * The scalar CPU backend accepts only the GPU-origin boundary; host-host is
   * its default. The AVX2 comparator runs host-host only.
   */
  if (is_cpu_backend(backend) &&
      (block_size_seen || grid_size_seen || k1_seen ||
       (boundary_seen && strcmp(boundary_name, "gpu-origin") != 0) ||
       (boundary_seen && strcmp(backend, "cpu-avx2") == 0))) {
    return SQ_ERR_ARGUMENT;
  }
  if (is_cpu_backend(backend) && !boundary_seen) {
    boundary_name = "host-host";
  }
  transfers = strcmp(boundary_name, "host-origin") == 0 ||
              strcmp(boundary_name, "gpu-origin") == 0;
  if (policy_seen && !transfers) {
    return SQ_ERR_ARGUMENT;
  }
  if (!transfers) {
    transfer_policy_name = "none";
  }
#ifndef SQ_ENABLE_CUDA
  if (strcmp(backend, "cuda") == 0 ||
      strcmp(boundary_name, "gpu-origin") == 0) {
    return SQ_ERR_CUDA_UNAVAILABLE;
  }
#endif
  status =
      resolve_cpu_threads(backend, threads_seen, requested_threads, &threads);
  if (status != SQ_OK) {
    return status;
  }
  if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX ||
      warmups > UINT64_MAX - reps) {
    return SQ_ERR_ID_OVERFLOW;
  }
  total_runs = warmups + reps;
  if (total_runs == 0 ||
      total_runs - 1 > (uint64_t)UINT32_MAX - invocation_id) {
    return SQ_ERR_ID_OVERFLOW;
  }
  if (reps > (uint64_t)(SIZE_MAX / sizeof *samples)) {
    return SQ_ERR_MEMORY;
  }

  if (!read_file(input_path, &input_bytes, &input_size)) {
    return SQ_ERR_IO;
  }
  if (input_size % sizeof(uint32_t) != 0) {
    status = SQ_ERR_PAYLOAD_LENGTH;
    goto done;
  }
  count = input_size / sizeof(uint32_t);
  if (count > SIZE_MAX / sizeof *values || count > SIZE_MAX - SQ_HEADER_SIZE) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  payload_bytes = bits == SQ_Q4_BITS ? count / 2 + (count & 1) : count;
  if (payload_bytes > SIZE_MAX - SQ_HEADER_SIZE) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  record_size = SQ_HEADER_SIZE + payload_bytes;
  values = count == 0 ? NULL : (float *)malloc(count * sizeof *values);
  words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
  generated_words =
      count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *generated_words);
  record = (uint8_t *)malloc(record_size);
  samples = (sq_bench_sample *)calloc((size_t)reps, sizeof *samples);
  scale_partial_count = sq_scale_workspace_elements(count);
  if (scale_partial_count == SIZE_MAX ||
      scale_partial_count > SIZE_MAX / sizeof *scale_partials) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  if (scale_partial_count != 0) {
    scale_partials =
        (float *)malloc(scale_partial_count * sizeof *scale_partials);
  }
  if ((count != 0 &&
       (values == NULL || words == NULL || generated_words == NULL)) ||
      record == NULL || samples == NULL ||
      (scale_partial_count != 0 && scale_partials == NULL)) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  for (size_t element = 0; element < count; element++) {
    uint32_t bits32 = sq_load_u32_le(input_bytes + 4 * element);
    memcpy(&values[element], &bits32, sizeof bits32);
  }
  if (words_path != NULL) {
    if (!read_file(words_path, &word_bytes, &word_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (count > SIZE_MAX / sizeof(uint32_t) ||
        word_size != count * sizeof(uint32_t)) {
      status = SQ_ERR_PAYLOAD_LENGTH;
      goto done;
    }
    for (size_t element = 0; element < count; element++) {
      words[element] = sq_load_u32_le(word_bytes + 4 * element);
    }
  }
  status = sq_validate_input(values, count);
  if (status != SQ_OK) {
    goto done;
  }
  if (scale_seen && (!isfinite(prescribed_scale) || prescribed_scale < 0.0f)) {
    status = SQ_ERR_SCALE;
    goto done;
  }

  cpu_gpu_origin =
      strcmp(backend, "cpu") == 0 && strcmp(boundary_name, "gpu-origin") == 0;
  if (is_cpu_backend(backend) && !cpu_gpu_origin) {
    if (threads != 0) {
      team_size = sq_avx2_team_size(threads);
    }
    status = bench_cpu_compress_one(
        (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
        scale_seen, prescribed_scale, words_path != NULL ? words : NULL,
        generated_words, scale_partials, scale_partial_count, threads, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (output_path != NULL && !write_file(output_path, record, record_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (!bench_clock_init(&clock_frequency)) {
      status = SQ_ERR_CLOCK;
      goto done;
    }
    for (uint64_t run = 0; run < total_runs; run++) {
      double start_ms, stop_ms;
      uint64_t run_offset = run < warmups ? reps + run : run - warmups;
      uint64_t current_invocation = invocation_id + run_offset;
      if (!bench_clock_now_ms(&clock_frequency, &start_ms)) {
        status = SQ_ERR_CLOCK;
        goto done;
      }
      status = bench_cpu_compress_one(
          (uint8_t)bits, values, count, seed, tensor_id, current_invocation,
          scale_seen, prescribed_scale, words_path != NULL ? words : NULL,
          generated_words, scale_partials, scale_partial_count, threads,
          record);
      if (status != SQ_OK) {
        goto done;
      }
      if (!bench_clock_now_ms(&clock_frequency, &stop_ms)) {
        status = SQ_ERR_CLOCK;
        goto done;
      }
      if (run >= warmups) {
        samples[run - warmups].wall_ms = stop_ms - start_ms;
      }
    }
  }
  // CPU-only builds end the chain above; the GPU branches continue it.
#ifdef SQ_ENABLE_CUDA
  else if (cpu_gpu_origin) {
    status = bench_cpu_gpu_origin(
        (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
        scale_seen, prescribed_scale, words_path != NULL ? words : NULL,
        generated_words, scale_partials, scale_partial_count,
        strcmp(transfer_policy_name, "pinned") == 0 ? SQ_CUDA_TRANSFER_PINNED
                                                    : SQ_CUDA_TRANSFER_PAGEABLE,
        warmups, reps, record, output_path, record_size, samples);
    if (status != SQ_OK) {
      goto done;
    }
  } else {
    uint8_t *base_payload = record + SQ_HEADER_SIZE;
    if (sq_cuda_select_k1(k1) != 0) {
      status = SQ_ERR_ARGUMENT;
      goto done;
    }
    status = sq_cuda_bench(
        (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
        scale_seen, prescribed_scale, words_path != NULL ? words : NULL,
        (int)block_size, (int)grid_size,
        strcmp(boundary_name, "resident") == 0 ? SQ_CUDA_BENCH_RESIDENT
        : strcmp(boundary_name, "resident-graph") == 0
            ? SQ_CUDA_BENCH_RESIDENT_GRAPH
        : strcmp(boundary_name, "gpu-origin") == 0 ? SQ_CUDA_BENCH_GPU_ORIGIN
                                                   : SQ_CUDA_BENCH_HOST_ORIGIN,
        strcmp(transfer_policy_name, "pinned") == 0 ? SQ_CUDA_TRANSFER_PINNED
                                                    : SQ_CUDA_TRANSFER_PAGEABLE,
        warmups, reps, base_payload, &scale, samples, &capture_ms);
    if (status != SQ_OK) {
      goto done;
    }
    status = sq_header_encode((uint8_t)bits, (uint64_t)count, scale, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (output_path != NULL && !write_file(output_path, record, record_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
  }
#endif

  print_bench_json(backend, boundary_name, transfer_policy_name, (uint8_t)bits,
                   count, seed, tensor_id, invocation_id, warmups, reps,
                   scale_seen, prescribed_scale, (int)block_size,
                   (int)grid_size,
                   strcmp(backend, "cuda") == 0 ? k1_name(k1) : NULL, team_size,
                   payload_bytes, samples, capture_ms);
  status = SQ_OK;

done:
  free(samples);
  free(scale_partials);
  free(generated_words);
  free(words);
  free(record);
  free(values);
  free(word_bytes);
  free(input_bytes);
  return status;
}

static sq_status decompress_file(int argc, char **argv) {
  const char *input_path = NULL, *output_path = NULL;
  uint8_t *record = NULL, *output = NULL;
  float *values = NULL;
  size_t record_size = 0, count = 0, output_size, i;
  sq_status status;
  int argument;

  for (argument = 2; argument < argc; argument++) {
    const char *option = argv[argument];
    if (argument + 1 >= argc) {
      return SQ_ERR_ARGUMENT;
    }
    if (strcmp(option, "--input") == 0) {
      input_path = argv[++argument];
    } else if (strcmp(option, "--output") == 0) {
      output_path = argv[++argument];
    } else {
      return SQ_ERR_ARGUMENT;
    }
  }
  if (input_path == NULL || output_path == NULL) {
    return SQ_ERR_ARGUMENT;
  }
  if (!read_file(input_path, &record, &record_size)) {
    return SQ_ERR_IO;
  }
  status = sq_decode_record(record, record_size, &values, &count);
  if (status != SQ_OK) {
    goto done;
  }
  if (count > SIZE_MAX / sizeof(uint32_t)) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  output_size = count * sizeof(uint32_t);
  output = count == 0 ? NULL : (uint8_t *)malloc(output_size);
  if (count != 0 && output == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  for (i = 0; i < count; i++) {
    uint32_t bits32;
    memcpy(&bits32, &values[i], sizeof bits32);
    sq_store_u32_le(output + 4 * i, bits32);
  }
  if (!write_file(output_path, output, output_size)) {
    status = SQ_ERR_IO;
  }

done:
  free(output);
  free(values);
  free(record);
  return status;
}

static uint32_t f32_bits(float value) {
  uint32_t bits;

  memcpy(&bits, &value, sizeof bits);
  return bits;
}

static void store_f64_le(uint8_t bytes[8], double value) {
  uint64_t bits64;

  memcpy(&bits64, &value, sizeof bits64);
  sq_store_u32_le(bytes, (uint32_t)bits64);
  sq_store_u32_le(bytes + 4, (uint32_t)(bits64 >> 32));
}

/* Compresses one input under seeds seed_start..seed_start+seeds-1 with the
   computed scale, decodes each record, and writes per-element FP64 sums of the
   decoded values followed by per-element sums of their squares. */
static sq_status expect_file(int argc, char **argv) {
  const char *input_path = NULL, *output_path = NULL;
  const char *backend = "cpu";
  uint64_t bits = SQ_Q8_BITS, seeds = 0, seed_start = 1;
  uint64_t tensor_id = 0, invocation_id = 0, t;
  int seeds_seen = 0, i;
  uint8_t *input_bytes = NULL, *record = NULL, *output = NULL;
  float *values = NULL, *decoded = NULL;
  uint32_t *words = NULL;
  double *sums = NULL, *squares = NULL;
  size_t input_size = 0, count = 0, payload_size, i_size, decoded_count;
  size_t output_size;
  float scale = 0.0f;
  uint32_t scale_bits;
  sq_status status;

  for (i = 2; i < argc; i++) {
    const char *option = argv[i];
    const char *value;
    if (i + 1 >= argc) {
      return SQ_ERR_ARGUMENT;
    }
    value = argv[++i];
    if (strcmp(option, "--input") == 0) {
      input_path = value;
    } else if (strcmp(option, "--output") == 0) {
      output_path = value;
    } else if (strcmp(option, "--seeds") == 0) {
      seeds_seen = parse_u64(value, &seeds) && seeds != 0;
      if (!seeds_seen) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--seed-start") == 0) {
      if (!parse_u64(value, &seed_start)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--tensor-id") == 0) {
      if (!parse_u64(value, &tensor_id)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--invocation-id") == 0) {
      if (!parse_u64(value, &invocation_id)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--bits") == 0) {
      if (!parse_u64(value, &bits)) {
        return SQ_ERR_ARGUMENT;
      }
    } else if (strcmp(option, "--backend") == 0) {
      if (strcmp(value, "cpu") != 0 && strcmp(value, "cuda") != 0) {
        return SQ_ERR_ARGUMENT;
      }
      backend = value;
    } else {
      return SQ_ERR_ARGUMENT;
    }
  }
  if (input_path == NULL || output_path == NULL || !seeds_seen) {
    return SQ_ERR_ARGUMENT;
  }
  if (seed_start > UINT64_MAX - (seeds - 1)) {
    return SQ_ERR_ARGUMENT;
  }
  if (bits != SQ_Q4_BITS && bits != SQ_Q8_BITS) {
    return SQ_ERR_BIT_WIDTH;
  }
#ifndef SQ_ENABLE_CUDA
  if (strcmp(backend, "cuda") == 0) {
    return SQ_ERR_CUDA_UNAVAILABLE;
  }
#endif
  if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX) {
    return SQ_ERR_ID_OVERFLOW;
  }

  if (!read_file(input_path, &input_bytes, &input_size)) {
    return SQ_ERR_IO;
  }
  if (input_size == 0 || input_size % sizeof(uint32_t) != 0) {
    status = SQ_ERR_PAYLOAD_LENGTH;
    goto done;
  }
  count = input_size / sizeof(uint32_t);
  payload_size = (bits == SQ_Q4_BITS) ? (count + 1) / 2 : count;
  if (count > SIZE_MAX / (2 * sizeof(double)) ||
      count > SIZE_MAX - SQ_HEADER_SIZE) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  values = (float *)malloc(count * sizeof *values);
  if (strcmp(backend, "cpu") == 0) {
    words = (uint32_t *)malloc(count * sizeof *words);
  }
  record = (uint8_t *)malloc(SQ_HEADER_SIZE + payload_size);
  sums = (double *)calloc(count, sizeof *sums);
  squares = (double *)calloc(count, sizeof *squares);
  if (values == NULL || (strcmp(backend, "cpu") == 0 && words == NULL) ||
      record == NULL || sums == NULL || squares == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  for (i_size = 0; i_size < count; i_size++) {
    uint32_t bits32 = sq_load_u32_le(input_bytes + 4 * i_size);
    memcpy(&values[i_size], &bits32, sizeof bits32);
  }
  status = sq_compute_scale(values, count, &scale);
  if (status != SQ_OK) {
    goto done;
  }

  for (t = 0; t < seeds; t++) {
    uint64_t seed = seed_start + t;
    float record_scale = scale;

    if (strcmp(backend, "cuda") == 0) {
#ifdef SQ_ENABLE_CUDA
      sq_cuda_timings timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      status = sq_cuda_compress(
          (uint8_t)bits, values, count, seed, tensor_id, invocation_id, 0, 0.0f,
          NULL, 256, 0, record + SQ_HEADER_SIZE, &record_scale, 0, &timings);
      if (status != SQ_OK) {
        goto done;
      }
      if (f32_bits(record_scale) != f32_bits(scale)) {
        status = SQ_ERR_SCALE;
        goto done;
      }
#endif
    } else {
      sq_rng_stream stream;
      if (sq_rng_stream_init(&stream, seed, tensor_id, invocation_id) !=
          SQ_RNG_OK) {
        status = SQ_ERR_ID_OVERFLOW;
        goto done;
      }
      sq_rng_words_cpu(&stream, (uint64_t)count, words);
      status = sq_encode_payload((uint8_t)bits, values, count, scale, words,
                                 record + SQ_HEADER_SIZE);
      if (status != SQ_OK) {
        goto done;
      }
    }
    status =
        sq_header_encode((uint8_t)bits, (uint64_t)count, record_scale, record);
    if (status != SQ_OK) {
      goto done;
    }
    status = sq_decode_record(record, SQ_HEADER_SIZE + payload_size, &decoded,
                              &decoded_count);
    if (status != SQ_OK) {
      goto done;
    }
    if (decoded_count != count) {
      status = SQ_ERR_COUNT;
      goto done;
    }
    for (i_size = 0; i_size < count; i_size++) {
      double y = (double)decoded[i_size];
      sums[i_size] += y;
      squares[i_size] += y * y;
    }
    free(decoded);
    decoded = NULL;
  }

  output_size = 2 * count * sizeof(double);
  output = (uint8_t *)malloc(output_size);
  if (output == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  for (i_size = 0; i_size < count; i_size++) {
    store_f64_le(output + 8 * i_size, sums[i_size]);
    store_f64_le(output + 8 * (count + i_size), squares[i_size]);
  }
  if (!write_file(output_path, output, output_size)) {
    status = SQ_ERR_IO;
    goto done;
  }
  scale_bits = f32_bits(scale);
  printf("{\"backend\":\"%s\",\"bits\":%u,\"count\":%llu,\"seeds\":%llu,"
         "\"seed_start\":%llu,\"tensor_id\":%llu,\"invocation_id\":%llu,"
         "\"scale\":%.9g,\"scale_bits\":\"0x%08x\"}\n",
         backend, (unsigned)bits, (unsigned long long)count,
         (unsigned long long)seeds, (unsigned long long)seed_start,
         (unsigned long long)tensor_id, (unsigned long long)invocation_id,
         (double)scale, (unsigned)scale_bits);

done:
  free(output);
  free(decoded);
  free(squares);
  free(sums);
  free(record);
  free(words);
  free(values);
  free(input_bytes);
  return status;
}

int main(int argc, char **argv) {
  sq_status status;

  if (argc < 2 || strcmp(argv[1], "--help") == 0 ||
      strcmp(argv[1], "-h") == 0) {
    usage(argc < 2 ? stderr : stdout);
    return argc < 2 ? 2 : 0;
  }
  if (strcmp(argv[1], "compress") == 0) {
    status = compress_file(argc, argv);
  } else if (strcmp(argv[1], "bench") == 0) {
    status = bench_file(argc, argv);
  } else if (strcmp(argv[1], "decompress") == 0) {
    status = decompress_file(argc, argv);
  } else if (strcmp(argv[1], "expect") == 0) {
    status = expect_file(argc, argv);
  } else {
    usage(stderr);
    return 2;
  }
  if (status != SQ_OK) {
    (void)fprintf(stderr, "%s: %s\n", argv[1], sq_status_message(status));
    return 1;
  }
  return 0;
}
