#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "bench_report.h"
#include "byteorder.h"
#include "cli.h"
#include "codec.h"
#include "cpu_compress.h"
#include "quantizer.h"
#include "quantizer_avx2.h"
#include "run_invocation.h"
#ifdef SQ_ENABLE_CUDA
#include "quantizer_cuda.h"
#endif

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
#include "bench_clock.h"

#ifdef SQ_ENABLE_CUDA
#define SQ_HAS_CUDA 1
static sq_cuda_config make_cuda_config(const sq_options *options,
                                       uint8_t bit_width, const float *values,
                                       size_t count, uint64_t seed,
                                       const uint32_t *prescribed_words,
                                       uint64_t invocation_id_step);
#else
#define SQ_HAS_CUDA 0
#endif

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

/* Sets *threads to 0 for the scalar comparator, else to the requested AVX2 team
 * size. */
static sq_status resolve_cpu_threads(sq_backend backend, int threads_seen,
                                     uint64_t requested, int *threads) {
  *threads = 0;
  if (backend != SQ_BACKEND_CPU_AVX2) {
    return threads_seen ? SQ_ERR_ARGUMENT : SQ_OK;
  }
  if (!sq_avx2_is_supported()) {
    return SQ_ERR_AVX2_UNAVAILABLE;
  }
  *threads = threads_seen ? (int)requested : default_thread_count();
  return SQ_OK;
}

static sq_status compress_file(int argc, char **argv) {
  sq_options options;
  const char *input_path, *output_path, *words_path;
  const char *backend;
  uint64_t seed, tensor_id, invocation_id, bits;
  float prescribed_scale;
#ifdef SQ_ENABLE_CUDA
  float scale;
#endif
  int scale_seen, threads;
  uint8_t *input_bytes = NULL, *word_bytes = NULL, *record = NULL;
  float *values = NULL;
  uint32_t *words = NULL;
  size_t input_size = 0, word_size = 0, count;
  sq_cpu_compress_workspace cpu_workspace = {0};
  sq_status status;

  status = sq_cli_parse(SQ_COMMAND_COMPRESS, argc, argv, &options);
  if (status == SQ_OK) {
    status = sq_cli_validate(SQ_COMMAND_COMPRESS, &options, SQ_HAS_CUDA);
  }
  if (status != SQ_OK) {
    return status;
  }
  input_path = options.input_path;
  output_path = options.output_path;
  words_path = options.words_path;
  backend = sq_backend_name(options.backend);
  seed = options.seed;
  tensor_id = options.tensor_id;
  invocation_id = options.invocation_id;
  bits = options.bits;
  scale_seen = (options.seen & SQ_SEEN_SCALE) != 0;
  prescribed_scale = options.scale;
  status = resolve_cpu_threads(options.backend,
                               (options.seen & SQ_SEEN_THREADS) != 0,
                               options.threads, &threads);
  if (status != SQ_OK) {
    return status;
  }
  if (!sq_invocation_range_valid(tensor_id, invocation_id, 0, 1, 0)) {
    return SQ_ERR_ID_OVERFLOW;
  }

  if (!sq_read_file(input_path, &input_bytes, &input_size)) {
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
  sq_load_f32_array_le(values, input_bytes, count);

  if (strcmp(backend, "cuda") == 0) {
#ifdef SQ_ENABLE_CUDA
    sq_cuda_timings cuda_timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int timings_seen = (options.seen & SQ_SEEN_TIMINGS) != 0;
    size_t payload_size = sq_payload_size((uint8_t)bits, count);

    if (words_path != NULL) {
      if (!sq_read_file(words_path, &word_bytes, &word_size)) {
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
      sq_load_u32_array_le(words, word_bytes, count);
    }

    record = (uint8_t *)malloc(SQ_HEADER_SIZE + payload_size);
    if (record == NULL) {
      status = SQ_ERR_MEMORY;
      goto done;
    }
    {
      sq_cuda_config config =
          make_cuda_config(&options, (uint8_t)bits, values, count, seed,
                           words_path != NULL ? words : NULL, 1);
      status = sq_cuda_compress(&config, record + SQ_HEADER_SIZE, &scale,
                                timings_seen ? &cuda_timings : NULL);
    }
    if (status != SQ_OK) {
      goto done;
    }
    status = sq_header_encode((uint8_t)bits, (uint64_t)count, scale, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (!sq_write_file(output_path, record, SQ_HEADER_SIZE + payload_size)) {
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

  if (words_path != NULL) {
    if (!sq_read_file(words_path, &word_bytes, &word_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (count > SIZE_MAX / sizeof(uint32_t) ||
        word_size != count * sizeof(uint32_t)) {
      status = SQ_ERR_PAYLOAD_LENGTH;
      goto done;
    }
  }

  if (words_path != NULL) {
    words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
    if (count != 0 && words == NULL) {
      status = SQ_ERR_MEMORY;
      goto done;
    }
    sq_load_u32_array_le(words, word_bytes, count);
  }

  {
    size_t payload_size = sq_payload_size((uint8_t)bits, count);
    record = (uint8_t *)malloc(SQ_HEADER_SIZE + payload_size);
    if (record == NULL) {
      status = SQ_ERR_MEMORY;
      goto done;
    }
    status = sq_cpu_compress_workspace_reserve(&cpu_workspace, options.backend,
                                               threads, count);
    if (status != SQ_OK) {
      goto done;
    }
    status = sq_cpu_compress(
        options.backend, (uint8_t)bits, values, count, seed, tensor_id,
        invocation_id, scale_seen, prescribed_scale,
        words_path != NULL ? words : NULL, threads, &cpu_workspace, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (!sq_write_file(output_path, record, SQ_HEADER_SIZE + payload_size)) {
      status = SQ_ERR_IO;
    }
  }

done:
  sq_cpu_compress_workspace_destroy(&cpu_workspace);
  free(record);
  free(words);
  free(word_bytes);
  free(values);
  free(input_bytes);
  return status;
}

#ifdef SQ_ENABLE_CUDA
/* Boundaries without timed transfers carry SQ_TRANSFER_NONE; the driver
   still takes a policy and expects pageable. */
static sq_cuda_transfer_policy cuda_transfer_policy(sq_transfer_policy policy) {
  return policy == SQ_TRANSFER_PINNED ? SQ_CUDA_TRANSFER_PINNED
                                      : SQ_CUDA_TRANSFER_PAGEABLE;
}

static sq_cuda_config make_cuda_config(const sq_options *options,
                                       uint8_t bit_width, const float *values,
                                       size_t count, uint64_t seed,
                                       const uint32_t *prescribed_words,
                                       uint64_t invocation_id_step) {
  sq_cuda_config config = {0};

  config.bit_width = bit_width;
  config.values = values;
  config.count = count;
  config.seed = seed;
  config.tensor_id = options->tensor_id;
  config.invocation_id = options->invocation_id;
  config.prescribed_scale_seen = (options->seen & SQ_SEEN_SCALE) != 0;
  config.prescribed_scale = options->scale;
  config.prescribed_words = prescribed_words;
  config.block_size = (int)options->block_size;
  config.grid_size = (int)options->grid_size;
  config.k1_variant = options->k1;
  config.boundary = (sq_cuda_bench_boundary)options->boundary;
  config.transfer_policy = cuda_transfer_policy(options->transfer_policy);
  config.warmups = options->warmups;
  config.reps = options->reps;
  config.invocation_id_step = invocation_id_step;
  return config;
}

/*
 * GPU-origin CPU path: the input starts on the device. Each run times one full
 * D2H into the landing buffer (d2h_ms), then CPU compression from it (cpu_ms).
 */
static sq_status bench_cpu_gpu_origin(
    sq_backend backend, uint8_t bits, const float *values, size_t count,
    uint64_t seed, uint64_t tensor_id, uint64_t invocation_id, int scale_seen,
    float prescribed_scale, const uint32_t *prescribed_words,
    sq_cpu_compress_workspace *workspace,
    sq_cuda_transfer_policy transfer_policy, uint64_t warmups, uint64_t reps,
    uint8_t *record, const char *output_path, size_t record_size,
    sq_bench_sample *samples, uint64_t invocation_id_step) {
  sq_cuda_staging *staging = NULL;
  bench_clock_frequency clock_frequency;
  const float *landing;
  double d2h_ms;
  int record_written = 0;
  sq_status status;

  status = sq_cuda_staging_create(values, count, transfer_policy, &staging);
  if (status != SQ_OK) {
    return status;
  }
  status = sq_cuda_staging_download(staging, &landing, &d2h_ms);
  if (status == SQ_OK) {
    status = sq_cpu_compress(backend, bits, landing, count, seed, tensor_id,
                             invocation_id, scale_seen, prescribed_scale,
                             prescribed_words, 0, workspace, record);
  }
  if (status != SQ_OK) {
    goto done;
  }
  if (output_path != NULL && !sq_write_file(output_path, record, record_size)) {
    status = SQ_ERR_IO;
    goto done;
  }
  record_written = output_path != NULL;
  if (!bench_clock_init(&clock_frequency)) {
    status = SQ_ERR_CLOCK;
    goto done;
  }
  for (uint64_t run = 0; run < warmups + reps; run++) {
    double start_ms, cpu_start_ms, stop_ms;
    uint64_t current_invocation = sq_run_invocation_id(
        invocation_id, warmups, reps, run, invocation_id_step);
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
    status = sq_cpu_compress(backend, bits, landing, count, seed, tensor_id,
                             current_invocation, scale_seen, prescribed_scale,
                             prescribed_words, 0, workspace, record);
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
  if (status != SQ_OK && record_written) {
    (void)remove(output_path);
  }
  sq_cuda_staging_destroy(staging);
  return status;
}
#endif

static sq_status bench_file(int argc, char **argv) {
  sq_options options;
  const sq_legality *legality;
  const char *input_path, *output_path, *words_path;
  double capture_ms = 0.0;
  uint64_t seed, tensor_id, invocation_id;
  uint64_t bits, block_size, grid_size;
  uint64_t warmups, reps, total_runs;
  float prescribed_scale;
#ifdef SQ_ENABLE_CUDA
  float scale;
#endif
  int scale_seen, threads, team_size = 0, cpu_gpu_origin;
  int k1;
  uint8_t *input_bytes = NULL, *word_bytes = NULL, *record = NULL;
  float *values = NULL;
  uint32_t *words = NULL;
  sq_cpu_compress_workspace cpu_workspace = {0};
  sq_bench_sample *samples = NULL;
  sq_bench_result result;
  size_t input_size = 0, word_size = 0, count = 0, payload_bytes;
  size_t record_size;
  bench_clock_frequency clock_frequency;
  sq_status status;

  status = sq_cli_parse(SQ_COMMAND_BENCH, argc, argv, &options);
  if (status == SQ_OK) {
    status = sq_cli_validate(SQ_COMMAND_BENCH, &options, SQ_HAS_CUDA);
  }
  if (status != SQ_OK) {
    return status;
  }
  legality = sq_cli_legality(options.backend, options.boundary);
  input_path = options.input_path;
  output_path = options.output_path;
  words_path = options.words_path;
  seed = options.seed;
  tensor_id = options.tensor_id;
  invocation_id = options.invocation_id;
  bits = options.bits;
  block_size = options.block_size;
  grid_size = options.grid_size;
  warmups = options.warmups;
  reps = options.reps;
  k1 = options.k1;
  scale_seen = (options.seen & SQ_SEEN_SCALE) != 0;
  prescribed_scale = options.scale;
  status = resolve_cpu_threads(options.backend,
                               (options.seen & SQ_SEEN_THREADS) != 0,
                               options.threads, &threads);
  if (status != SQ_OK) {
    return status;
  }

  if (!sq_invocation_range_valid(tensor_id, invocation_id, warmups, reps,
                                 legality->id_step)) {
    return SQ_ERR_ID_OVERFLOW;
  }
  total_runs = warmups + reps;
  if (reps > (uint64_t)(SIZE_MAX / sizeof *samples)) {
    return SQ_ERR_MEMORY;
  }

  if (!sq_read_file(input_path, &input_bytes, &input_size)) {
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
  payload_bytes = sq_payload_size((uint8_t)bits, count);
  if (payload_bytes > SIZE_MAX - SQ_HEADER_SIZE) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  record_size = SQ_HEADER_SIZE + payload_bytes;
  values = count == 0 ? NULL : (float *)malloc(count * sizeof *values);
  words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
  record = (uint8_t *)malloc(record_size);
  samples = (sq_bench_sample *)calloc((size_t)reps, sizeof *samples);
  if ((count != 0 && (values == NULL || words == NULL)) || record == NULL ||
      samples == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  sq_load_f32_array_le(values, input_bytes, count);
  if (words_path != NULL) {
    if (!sq_read_file(words_path, &word_bytes, &word_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (count > SIZE_MAX / sizeof(uint32_t) ||
        word_size != count * sizeof(uint32_t)) {
      status = SQ_ERR_PAYLOAD_LENGTH;
      goto done;
    }
    sq_load_u32_array_le(words, word_bytes, count);
  }
  status = sq_validate_input(values, count);
  if (status != SQ_OK) {
    goto done;
  }
  if (scale_seen && (!isfinite(prescribed_scale) || prescribed_scale < 0.0f)) {
    status = SQ_ERR_SCALE;
    goto done;
  }

  cpu_gpu_origin = options.backend == SQ_BACKEND_CPU &&
                   options.boundary == SQ_BOUNDARY_GPU_ORIGIN;
  if (options.backend != SQ_BACKEND_CUDA) {
    status = sq_cpu_compress_workspace_reserve(&cpu_workspace, options.backend,
                                               threads, count);
    if (status != SQ_OK) {
      goto done;
    }
  }
  if (options.backend != SQ_BACKEND_CUDA && !cpu_gpu_origin) {
    if (threads != 0) {
      team_size = sq_avx2_team_size(threads);
    }
    status = sq_cpu_compress(
        options.backend, (uint8_t)bits, values, count, seed, tensor_id,
        invocation_id, scale_seen, prescribed_scale,
        words_path != NULL ? words : NULL, threads, &cpu_workspace, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (output_path != NULL &&
        !sq_write_file(output_path, record, record_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
    if (!bench_clock_init(&clock_frequency)) {
      status = SQ_ERR_CLOCK;
      goto done;
    }
    for (uint64_t run = 0; run < total_runs; run++) {
      double start_ms, stop_ms;
      uint64_t current_invocation = sq_run_invocation_id(
          invocation_id, warmups, reps, run, legality->id_step);
      if (!bench_clock_now_ms(&clock_frequency, &start_ms)) {
        status = SQ_ERR_CLOCK;
        goto done;
      }
      status = sq_cpu_compress(
          options.backend, (uint8_t)bits, values, count, seed, tensor_id,
          current_invocation, scale_seen, prescribed_scale,
          words_path != NULL ? words : NULL, threads, &cpu_workspace, record);
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
        options.backend, (uint8_t)bits, values, count, seed, tensor_id,
        invocation_id, scale_seen, prescribed_scale,
        words_path != NULL ? words : NULL, &cpu_workspace,
        cuda_transfer_policy(options.transfer_policy), warmups, reps, record,
        output_path, record_size, samples, legality->id_step);
    if (status != SQ_OK) {
      goto done;
    }
  } else {
    uint8_t *base_payload = record + SQ_HEADER_SIZE;
    sq_cuda_config config =
        make_cuda_config(&options, (uint8_t)bits, values, count, seed,
                         words_path != NULL ? words : NULL, legality->id_step);
    status = sq_cuda_bench(&config, base_payload, &scale, samples, &capture_ms);
    if (status != SQ_OK) {
      goto done;
    }
    status = sq_header_encode((uint8_t)bits, (uint64_t)count, scale, record);
    if (status != SQ_OK) {
      goto done;
    }
    if (output_path != NULL &&
        !sq_write_file(output_path, record, record_size)) {
      status = SQ_ERR_IO;
      goto done;
    }
  }
#endif

  result.backend = sq_backend_name(options.backend);
  result.boundary = legality->name;
  result.transfer_policy = sq_transfer_policy_name(options.transfer_policy);
  result.k1 = options.backend == SQ_BACKEND_CUDA ? sq_k1_name(k1) : NULL;
  result.columns = legality->columns;
  result.invocation_id_step = legality->id_step;
  result.bit_width = (uint8_t)bits;
  result.count = count;
  result.seed = seed;
  result.tensor_id = tensor_id;
  result.invocation_id = invocation_id;
  result.warmups = warmups;
  result.reps = reps;
  result.prescribed_scale_seen = scale_seen;
  result.prescribed_scale = prescribed_scale;
  result.block_size = (int)block_size;
  result.grid_size = (int)grid_size;
  result.team_size = team_size;
  result.payload_bytes = payload_bytes;
  result.samples = samples;
  result.capture_ms = capture_ms;
  sq_bench_print_json(&result);
  status = SQ_OK;

done:
  free(samples);
  sq_cpu_compress_workspace_destroy(&cpu_workspace);
  free(words);
  free(record);
  free(values);
  free(word_bytes);
  free(input_bytes);
  return status;
}

static sq_status decompress_file(int argc, char **argv) {
  sq_options options;
  const char *input_path, *output_path;
  uint8_t *record = NULL, *output = NULL;
  float *values = NULL;
  size_t record_size = 0, count = 0, output_size;
  sq_status status;

  status = sq_cli_parse(SQ_COMMAND_DECOMPRESS, argc, argv, &options);
  if (status == SQ_OK) {
    status = sq_cli_validate(SQ_COMMAND_DECOMPRESS, &options, SQ_HAS_CUDA);
  }
  if (status != SQ_OK) {
    return status;
  }
  input_path = options.input_path;
  output_path = options.output_path;

  if (!sq_read_file(input_path, &record, &record_size)) {
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
  sq_store_f32_array_le(output, values, count);
  if (!sq_write_file(output_path, output, output_size)) {
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
  sq_store_u64_le(bytes, bits64);
}

/* Compresses one input under seeds seed_start..seed_start+seeds-1 with the
   computed scale, decodes each record, and writes per-element FP64 sums of the
   decoded values followed by per-element sums of their squares. */
static sq_status expect_file(int argc, char **argv) {
  sq_options options;
  const char *input_path, *output_path, *backend;
  uint64_t bits, seeds, seed_start, tensor_id, invocation_id, t;
  uint8_t *input_bytes = NULL, *record = NULL, *output = NULL;
  float *values = NULL, *decoded = NULL;
  double *sums = NULL, *squares = NULL;
  sq_cpu_compress_workspace cpu_workspace = {0};
  size_t input_size = 0, count = 0, payload_size, i_size, decoded_count;
  size_t output_size;
  float scale = 0.0f;
  uint32_t scale_bits;
  sq_status status;

  status = sq_cli_parse(SQ_COMMAND_EXPECT, argc, argv, &options);
  if (status == SQ_OK) {
    status = sq_cli_validate(SQ_COMMAND_EXPECT, &options, SQ_HAS_CUDA);
  }
  if (status != SQ_OK) {
    return status;
  }
  input_path = options.input_path;
  output_path = options.output_path;
  backend = sq_backend_name(options.backend);
  bits = options.bits;
  seeds = options.seeds;
  seed_start = options.seed_start;
  tensor_id = options.tensor_id;
  invocation_id = options.invocation_id;
  if (!sq_invocation_range_valid(tensor_id, invocation_id, 0, 1, 0)) {
    return SQ_ERR_ID_OVERFLOW;
  }

  if (!sq_read_file(input_path, &input_bytes, &input_size)) {
    return SQ_ERR_IO;
  }
  if (input_size == 0 || input_size % sizeof(uint32_t) != 0) {
    status = SQ_ERR_PAYLOAD_LENGTH;
    goto done;
  }
  count = input_size / sizeof(uint32_t);
  payload_size = sq_payload_size((uint8_t)bits, count);
  if (count > SIZE_MAX / (2 * sizeof(double)) ||
      count > SIZE_MAX - SQ_HEADER_SIZE) {
    status = SQ_ERR_COUNT;
    goto done;
  }
  values = (float *)malloc(count * sizeof *values);
  record = (uint8_t *)malloc(SQ_HEADER_SIZE + payload_size);
  sums = (double *)calloc(count, sizeof *sums);
  squares = (double *)calloc(count, sizeof *squares);
  if (values == NULL || record == NULL || sums == NULL || squares == NULL) {
    status = SQ_ERR_MEMORY;
    goto done;
  }
  sq_load_f32_array_le(values, input_bytes, count);
  status = sq_compute_scale(values, count, &scale);
  if (status != SQ_OK) {
    goto done;
  }
  if (strcmp(backend, "cpu") == 0) {
    status = sq_cpu_compress_workspace_reserve(&cpu_workspace, SQ_BACKEND_CPU,
                                               0, count);
    if (status != SQ_OK) {
      goto done;
    }
  }

  for (t = 0; t < seeds; t++) {
    uint64_t seed = seed_start + t;
#ifdef SQ_ENABLE_CUDA
    float record_scale = scale;
#endif

    if (strcmp(backend, "cuda") == 0) {
#ifdef SQ_ENABLE_CUDA
      sq_cuda_config config = make_cuda_config(&options, (uint8_t)bits, values,
                                               count, seed, NULL, 1);
      status = sq_cuda_compress(&config, record + SQ_HEADER_SIZE, &record_scale,
                                NULL);
      if (status != SQ_OK) {
        goto done;
      }
      if (f32_bits(record_scale) != f32_bits(scale)) {
        status = SQ_ERR_SCALE;
        goto done;
      }
      status = sq_header_encode((uint8_t)bits, (uint64_t)count, record_scale,
                                record);
      if (status != SQ_OK) {
        goto done;
      }
#endif
    } else {
      status = sq_cpu_compress(SQ_BACKEND_CPU, (uint8_t)bits, values, count,
                               seed, tensor_id, invocation_id, 1, scale, NULL,
                               0, &cpu_workspace, record);
      if (status != SQ_OK) {
        goto done;
      }
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
  if (!sq_write_file(output_path, output, output_size)) {
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
  sq_cpu_compress_workspace_destroy(&cpu_workspace);
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
