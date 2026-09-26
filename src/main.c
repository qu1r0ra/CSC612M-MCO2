#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "codec.h"
#include "byteorder.h"
#include "quantizer.h"
#include "rng_cpu.h"
#include "quantizer_cuda.h"

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
#endif

static void usage(FILE *stream)
{
    fprintf(stream,
            "Usage:\n"
            "  mco2 compress --input INPUT.f32 --output RECORD --seed UINT64\n"
            "      [--backend cpu|cuda] [--bits 4|8] [--tensor-id UINT32]\n"
            "      [--invocation-id UINT32] [--scale FP32] [--words WORDS.u32]\n"
            "      [--block-size UINT32] [--grid-size UINT32] [--timings]\n"
            "  mco2 bench --input INPUT.f32 --seed UINT64\n"
            "      [--output RECORD | --record-output RECORD]\n"
            "      [--backend cpu|cuda] [--bits 4|8] [--tensor-id UINT32]\n"
            "      [--invocation-id UINT32] [--scale FP32] [--words WORDS.u32]\n"
            "      [--boundary resident|resident-graph|host-origin] [--warmup UINT32 (default 10)]\n"
            "      [--reps UINT32 (default 30)]\n"
            "      [--block-size UINT32] [--grid-size UINT32] [--timings]\n"
            "  mco2 decompress --input RECORD --output OUTPUT.f32\n"
            "\n"
            "Input and output tensors use little-endian raw FP32. Prescribed\n"
            "random words use little-endian raw uint32, one per element.\n");
}

static int read_file(const char *path, uint8_t **bytes, size_t *size)
{
    FILE *file;
    long length;
    uint8_t *buffer = NULL;
    size_t read_count;

    *bytes = NULL;
    *size = 0;
    file = fopen(path, "rb");
    if (file == NULL)
        return 0;
    if (fseek(file, 0, SEEK_END) != 0)
        goto fail;
    length = ftell(file);
    if (length < 0 || (uintmax_t)length > (uintmax_t)SIZE_MAX)
        goto fail;
    if (fseek(file, 0, SEEK_SET) != 0)
        goto fail;
    if (length != 0) {
        buffer = (uint8_t *)malloc((size_t)length);
        if (buffer == NULL)
            goto fail;
        read_count = fread(buffer, 1, (size_t)length, file);
        if (read_count != (size_t)length)
            goto fail;
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

static int write_file(const char *path, const uint8_t *bytes, size_t size)
{
    FILE *file = fopen(path, "wb");
    int ok;

    if (file == NULL)
        return 0;
    ok = size == 0 || fwrite(bytes, 1, size, file) == size;
    if (fclose(file) != 0)
        ok = 0;
    return ok;
}

static int parse_u64(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;

    if (text[0] == '\0' || text[0] == '-')
        return 0;
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno == ERANGE || end == text || *end != '\0')
        return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int parse_scale(const char *text, float *scale)
{
    char *end;
    float parsed;

    errno = 0;
    parsed = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(parsed) ||
        parsed < 0.0f)
        return 0;
    *scale = parsed;
    return 1;
}

static mco2_q8_status compress_file(int argc, char **argv)
{
    const char *input_path = NULL, *output_path = NULL, *words_path = NULL;
    const char *backend = "cpu";
    uint64_t seed = 0, tensor_id = 0, invocation_id = 0, bits = MCO2_Q8_BITS;
    uint64_t block_size = 256, grid_size = 0;
    float prescribed_scale = 0.0f, scale;
    int seed_seen = 0, scale_seen = 0, timings_seen = 0;
    int block_size_seen = 0, grid_size_seen = 0, i;
    uint8_t *input_bytes = NULL, *word_bytes = NULL, *record = NULL;
    float *values = NULL;
    uint32_t *words = NULL;
    uint8_t *codes;
    size_t input_size = 0, word_size = 0, count, i_size;
    mco2_rng_stream stream;
    mco2_q8_status status;

    for (i = 2; i < argc; i++) {
        const char *option = argv[i];
        const char *value;
        if (strcmp(option, "--timings") == 0) {
            timings_seen = 1;
            continue;
        }
        if (i + 1 >= argc)
            return MCO2_Q8_ERR_ARGUMENT;
        value = argv[++i];
        if (strcmp(option, "--input") == 0)
            input_path = value;
        else if (strcmp(option, "--output") == 0)
            output_path = value;
        else if (strcmp(option, "--words") == 0)
            words_path = value;
        else if (strcmp(option, "--seed") == 0) {
            seed_seen = parse_u64(value, &seed);
            if (!seed_seen)
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--tensor-id") == 0) {
            if (!parse_u64(value, &tensor_id))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--invocation-id") == 0) {
            if (!parse_u64(value, &invocation_id))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--bits") == 0) {
            if (!parse_u64(value, &bits))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--scale") == 0) {
            scale_seen = parse_scale(value, &prescribed_scale);
            if (!scale_seen)
                return MCO2_Q8_ERR_SCALE;
        } else if (strcmp(option, "--backend") == 0) {
            if (strcmp(value, "cpu") != 0 && strcmp(value, "cuda") != 0)
                return MCO2_Q8_ERR_ARGUMENT;
            backend = value;
        } else if (strcmp(option, "--block-size") == 0) {
            if (!parse_u64(value, &block_size) || block_size == 0 ||
                block_size > 1024)
                return MCO2_Q8_ERR_ARGUMENT;
            block_size_seen = 1;
        } else if (strcmp(option, "--grid-size") == 0) {
            if (!parse_u64(value, &grid_size) || grid_size == 0 ||
                grid_size > 65535)
                return MCO2_Q8_ERR_ARGUMENT;
            grid_size_seen = 1;
        } else {
            return MCO2_Q8_ERR_ARGUMENT;
        }
    }
    if (input_path == NULL || output_path == NULL || !seed_seen)
        return MCO2_Q8_ERR_ARGUMENT;
    if (bits != MCO2_Q4_BITS && bits != MCO2_Q8_BITS)
        return MCO2_Q8_ERR_BIT_WIDTH;
    if ((block_size_seen || grid_size_seen) && strcmp(backend, "cuda") != 0)
        return MCO2_Q8_ERR_ARGUMENT;
    if (timings_seen && strcmp(backend, "cuda") != 0)
        return MCO2_Q8_ERR_TIMINGS_BACKEND;
#ifndef MCO2_ENABLE_CUDA
    if (strcmp(backend, "cuda") == 0)
        return MCO2_Q8_ERR_CUDA_UNAVAILABLE;
#endif
    if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX)
        return MCO2_Q8_ERR_ID_OVERFLOW;

    if (!read_file(input_path, &input_bytes, &input_size))
        return MCO2_Q8_ERR_IO;
    if (input_size % sizeof(uint32_t) != 0) {
        status = MCO2_Q8_ERR_PAYLOAD_LENGTH;
        goto done;
    }
    count = input_size / sizeof(uint32_t);
    if (count > SIZE_MAX / sizeof *values ||
        count > SIZE_MAX - MCO2_Q8_HEADER_SIZE) {
        status = MCO2_Q8_ERR_COUNT;
        goto done;
    }
    values = count == 0 ? NULL : (float *)malloc(count * sizeof *values);
    if (count != 0 && values == NULL) {
        status = MCO2_Q8_ERR_MEMORY;
        goto done;
    }
    for (i_size = 0; i_size < count; i_size++) {
        uint32_t bits32 = mco2_load_u32_le(input_bytes + 4 * i_size);
        memcpy(&values[i_size], &bits32, sizeof bits32);
    }

    if (strcmp(backend, "cuda") == 0) {
#ifdef MCO2_ENABLE_CUDA
        mco2_cuda_timings cuda_timings = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        size_t payload_size = (bits == MCO2_Q4_BITS) ? (count + 1) / 2 : count;

        if (words_path != NULL) {
            if (!read_file(words_path, &word_bytes, &word_size)) {
                status = MCO2_Q8_ERR_IO;
                goto done;
            }
            if (count > SIZE_MAX / sizeof(uint32_t) ||
                word_size != count * sizeof(uint32_t)) {
                status = MCO2_Q8_ERR_PAYLOAD_LENGTH;
                goto done;
            }
            words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
            if (count != 0 && words == NULL) {
                status = MCO2_Q8_ERR_MEMORY;
                goto done;
            }
            for (i_size = 0; i_size < count; i_size++)
                words[i_size] = mco2_load_u32_le(word_bytes + 4 * i_size);
        }

        record = (uint8_t *)malloc(MCO2_Q8_HEADER_SIZE + payload_size);
        if (record == NULL) {
            status = MCO2_Q8_ERR_MEMORY;
            goto done;
        }
        status = mco2_cuda_compress(
            (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
            scale_seen, prescribed_scale, words, (int)block_size,
            (int)grid_size, record + MCO2_Q8_HEADER_SIZE, &scale,
            timings_seen, &cuda_timings);
        if (status != MCO2_Q8_OK)
            goto done;
        status = mco2_q8_header_encode((uint8_t)bits, (uint64_t)count,
                                       scale, record);
        if (status != MCO2_Q8_OK)
            goto done;
        if (!write_file(output_path, record,
                        MCO2_Q8_HEADER_SIZE + payload_size)) {
            status = MCO2_Q8_ERR_IO;
            goto done;
        }
        if (timings_seen) {
            fprintf(stderr,
                    "{\"k1_ms\":%.6f,\"k2_ms\":%.6f,\"k3_ms\":%.6f,"
                    "\"h2d_ms\":%.6f,\"d2h_ms\":%.6f}\n",
                    cuda_timings.k1_ms, cuda_timings.k2_ms, cuda_timings.k3_ms,
                    cuda_timings.h2d_ms, cuda_timings.d2h_ms);
        }
        status = MCO2_Q8_OK;
        goto done;
#endif
    }

    if (scale_seen) {
        scale = prescribed_scale;
    } else {
        status = mco2_q8_compute_scale(values, count, &scale);
        if (status != MCO2_Q8_OK)
            goto done;
    }

    if (words_path != NULL) {
        if (!read_file(words_path, &word_bytes, &word_size)) {
            status = MCO2_Q8_ERR_IO;
            goto done;
        }
        if (count > SIZE_MAX / sizeof(uint32_t) ||
            word_size != count * sizeof(uint32_t)) {
            status = MCO2_Q8_ERR_PAYLOAD_LENGTH;
            goto done;
        }
    }

    words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
    if (count != 0 && words == NULL) {
        status = MCO2_Q8_ERR_MEMORY;
        goto done;
    }
    if (words_path != NULL) {
        for (i_size = 0; i_size < count; i_size++)
            words[i_size] = mco2_load_u32_le(word_bytes + 4 * i_size);
    } else {
        if (mco2_rng_stream_init(&stream, seed, tensor_id, invocation_id) != MCO2_OK) {
            status = MCO2_Q8_ERR_ID_OVERFLOW;
            goto done;
        }
        mco2_rng_words_cpu(&stream, (uint64_t)count, words);
    }

    {
        size_t payload_size = (bits == MCO2_Q4_BITS) ? (count + 1) / 2 : count;
        record = (uint8_t *)malloc(MCO2_Q8_HEADER_SIZE + payload_size);
        if (record == NULL) {
            status = MCO2_Q8_ERR_MEMORY;
            goto done;
        }
        status = mco2_q8_header_encode((uint8_t)bits, (uint64_t)count, scale, record);
        if (status != MCO2_Q8_OK)
            goto done;
        codes = record + MCO2_Q8_HEADER_SIZE;
        status = mco2_encode_payload((uint8_t)bits, values, count, scale, words, codes);
        if (status != MCO2_Q8_OK)
            goto done;
        if (!write_file(output_path, record, MCO2_Q8_HEADER_SIZE + payload_size))
            status = MCO2_Q8_ERR_IO;
    }

done:
    free(record);
    free(words);
    free(word_bytes);
    free(values);
    free(input_bytes);
    return status;
}

#ifdef _WIN32
typedef LARGE_INTEGER bench_clock_frequency;
static int bench_clock_init(bench_clock_frequency *frequency)
{
    return QueryPerformanceFrequency(frequency) != 0;
}
static int bench_clock_now_ms(const bench_clock_frequency *frequency,
                              double *milliseconds)
{
    LARGE_INTEGER counter;
    if (QueryPerformanceCounter(&counter) == 0)
        return 0;
    *milliseconds = (double)counter.QuadPart * 1000.0 /
                    (double)frequency->QuadPart;
    return 1;
}
#else
typedef int bench_clock_frequency;
static int bench_clock_init(bench_clock_frequency *frequency)
{
    (void)frequency;
    return 1;
}
static int bench_clock_now_ms(const bench_clock_frequency *frequency,
                              double *milliseconds)
{
    struct timespec current;
    (void)frequency;
    if (clock_gettime(CLOCK_MONOTONIC, &current) != 0)
        return 0;
    *milliseconds = (double)current.tv_sec * 1000.0 +
                    (double)current.tv_nsec / 1000000.0;
    return 1;
}
#endif

static mco2_q8_status bench_cpu_compress_one(
    uint8_t bit_width, const float *values, size_t count, uint64_t seed,
    uint64_t tensor_id, uint64_t invocation_id, int prescribed_scale_seen,
    float prescribed_scale, const uint32_t *prescribed_words,
    uint32_t *generated_words, float *scale_partials,
    size_t scale_partial_capacity, uint8_t *record)
{
    float scale;
    mco2_rng_stream stream;
    mco2_q8_status status;
    if (prescribed_scale_seen) {
        scale = prescribed_scale;
    } else {
        status = mco2_q8_compute_scale_with_workspace(
            values, count, &scale, scale_partials, scale_partial_capacity);
        if (status != MCO2_Q8_OK)
            return status;
    }
    if (prescribed_words == NULL) {
        status = mco2_rng_stream_init(&stream, seed, tensor_id, invocation_id);
        if (status != MCO2_Q8_OK)
            return MCO2_Q8_ERR_ID_OVERFLOW;
        mco2_rng_words_cpu(&stream, (uint64_t)count, generated_words);
    }
    status = mco2_q8_header_encode(bit_width, (uint64_t)count, scale, record);
    if (status != MCO2_Q8_OK)
        return status;
    return mco2_encode_payload(
        bit_width, values, count, scale,
        prescribed_words != NULL ? prescribed_words : generated_words,
        record + MCO2_Q8_HEADER_SIZE);
}

typedef enum {
    BENCH_SAMPLE_WALL_TIME,
    BENCH_SAMPLE_K1_TIME,
    BENCH_SAMPLE_K2_TIME,
    BENCH_SAMPLE_K3_TIME,
    BENCH_SAMPLE_H2D_TIME,
    BENCH_SAMPLE_D2H_TIME
} mco2_bench_timing_field;

static void print_double_array(const mco2_bench_sample *samples,
                               uint64_t reps,
                               mco2_bench_timing_field field)
{
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
        default:
            value = 0.0;
            break;
        }
        if (i != 0)
            putchar(',');
        printf("%.9f", value);
    }
    putchar(']');
}

static void print_invocation_ids(uint64_t base_invocation_id,
                                 uint64_t count, uint64_t offset)
{
    uint64_t i;
    putchar('[');
    for (i = 0; i < count; i++) {
        if (i != 0)
            putchar(',');
        printf("%llu", (unsigned long long)(base_invocation_id + offset + i));
    }
    putchar(']');
}

static void print_bench_json(
    const char *backend, const char *boundary, uint8_t bit_width,
    size_t count, uint64_t seed, uint64_t tensor_id,
    uint64_t invocation_id, uint64_t warmups, uint64_t reps,
    int prescribed_scale_seen, float prescribed_scale, int block_size,
    int grid_size, size_t payload_bytes, const mco2_bench_sample *samples,
    double capture_ms)
{
    printf("{\"configuration\":{\"backend\":\"%s\",\"bits\":%u,"
           "\"count\":%llu,\"seed\":%llu,\"tensor_id\":%llu,"
           "\"invocation_id\":%llu,\"warmup\":%llu,\"reps\":%llu,"
           "\"repetition_invocation_ids\":",
           backend, (unsigned int)bit_width, (unsigned long long)count,
           (unsigned long long)seed, (unsigned long long)tensor_id,
           (unsigned long long)invocation_id, (unsigned long long)warmups,
           (unsigned long long)reps);
    print_invocation_ids(invocation_id, reps, 0);
    printf(",\"warmup_invocation_ids\":");
    print_invocation_ids(invocation_id, warmups, reps);
    printf(",\"boundary\":\"%s\",\"block_size\":%d,\"grid_size\":%d,"
           "\"prescribed_scale\":", boundary, block_size, grid_size);
    if (prescribed_scale_seen)
        printf("%.9g", (double)prescribed_scale);
    else
        printf("null");
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
                printf(",\"d2h_ms\":");
                print_double_array(samples, reps, BENCH_SAMPLE_D2H_TIME);
            }
        } else {
            printf(",\"capture_ms\":%.6f,\"capture_and_instantiate_ms\":%.6f",
                   capture_ms, capture_ms);
        }
    }
    printf(",\"header_bytes\":%d,\"payload_bytes\":%llu}\n",
           MCO2_Q8_HEADER_SIZE, (unsigned long long)payload_bytes);
}

static mco2_q8_status bench_file(int argc, char **argv)
{
    const char *input_path = NULL, *output_path = NULL, *words_path = NULL;
    const char *backend = "cpu", *boundary_name = "host-origin";
    double capture_ms = 0.0;
    uint64_t seed = 0, tensor_id = 0, invocation_id = 0;
    uint64_t bits = MCO2_Q8_BITS, block_size = 256, grid_size = 0;
    uint64_t warmups = 10, reps = 30, total_runs;
    float prescribed_scale = 0.0f;
#ifdef MCO2_ENABLE_CUDA
    float scale;
#endif
    int seed_seen = 0, scale_seen = 0, boundary_seen = 0;
    int block_size_seen = 0, grid_size_seen = 0, i;
    uint8_t *input_bytes = NULL, *word_bytes = NULL, *record = NULL;
    float *values = NULL, *scale_partials = NULL;
    uint32_t *words = NULL, *generated_words = NULL;
    mco2_bench_sample *samples = NULL;
    size_t input_size = 0, word_size = 0, count = 0, payload_bytes;
    size_t record_size, scale_partial_count;
    bench_clock_frequency clock_frequency;
    mco2_q8_status status = MCO2_Q8_ERR_ARGUMENT;

    for (i = 2; i < argc; i++) {
        const char *option = argv[i];
        const char *value;

        if (strcmp(option, "--timings") == 0)
            continue;
        if (i + 1 >= argc)
            return MCO2_Q8_ERR_ARGUMENT;
        value = argv[++i];
        if (strcmp(option, "--input") == 0)
            input_path = value;
        else if (strcmp(option, "--output") == 0 ||
                 strcmp(option, "--record-output") == 0)
            output_path = value;
        else if (strcmp(option, "--words") == 0)
            words_path = value;
        else if (strcmp(option, "--seed") == 0) {
            seed_seen = parse_u64(value, &seed);
            if (!seed_seen)
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--tensor-id") == 0) {
            if (!parse_u64(value, &tensor_id))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--invocation-id") == 0) {
            if (!parse_u64(value, &invocation_id))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--bits") == 0) {
            if (!parse_u64(value, &bits))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--scale") == 0) {
            scale_seen = parse_scale(value, &prescribed_scale);
            if (!scale_seen)
                return MCO2_Q8_ERR_SCALE;
        } else if (strcmp(option, "--backend") == 0) {
            if (strcmp(value, "cpu") != 0 && strcmp(value, "cuda") != 0)
                return MCO2_Q8_ERR_ARGUMENT;
            backend = value;
        } else if (strcmp(option, "--boundary") == 0) {
            if (strcmp(value, "resident") != 0 &&
                strcmp(value, "resident-graph") != 0 &&
                strcmp(value, "host-origin") != 0)
                return MCO2_Q8_ERR_ARGUMENT;
            boundary_name = value;
            boundary_seen = 1;
        } else if (strcmp(option, "--warmup") == 0) {
            if (!parse_u64(value, &warmups))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--reps") == 0) {
            if (!parse_u64(value, &reps))
                return MCO2_Q8_ERR_ARGUMENT;
        } else if (strcmp(option, "--block-size") == 0) {
            if (!parse_u64(value, &block_size) || block_size == 0 ||
                block_size > MCO2_CUDA_MAX_BLOCK_SIZE)
                return MCO2_Q8_ERR_ARGUMENT;
            block_size_seen = 1;
        } else if (strcmp(option, "--grid-size") == 0) {
            if (!parse_u64(value, &grid_size) || grid_size == 0 ||
                grid_size > MCO2_CUDA_MAX_GRID_SIZE)
                return MCO2_Q8_ERR_ARGUMENT;
            grid_size_seen = 1;
        } else {
            return MCO2_Q8_ERR_ARGUMENT;
        }
    }

    if (input_path == NULL || !seed_seen || reps == 0)
        return MCO2_Q8_ERR_ARGUMENT;
    if (bits != MCO2_Q4_BITS && bits != MCO2_Q8_BITS)
        return MCO2_Q8_ERR_BIT_WIDTH;
    if ((block_size_seen || grid_size_seen || boundary_seen) &&
        strcmp(backend, "cpu") == 0)
        return MCO2_Q8_ERR_ARGUMENT;
    if (strcmp(backend, "cuda") == 0 && !boundary_seen)
        boundary_name = "host-origin";
#ifndef MCO2_ENABLE_CUDA
    if (strcmp(backend, "cuda") == 0)
        return MCO2_Q8_ERR_CUDA_UNAVAILABLE;
#endif
    if (tensor_id > UINT32_MAX || invocation_id > UINT32_MAX ||
        warmups > UINT64_MAX - reps ||
        (total_runs = warmups + reps) == 0 ||
        total_runs - 1 > (uint64_t)UINT32_MAX - invocation_id)
        return MCO2_Q8_ERR_ID_OVERFLOW;
    if (reps > (uint64_t)(SIZE_MAX / sizeof *samples))
        return MCO2_Q8_ERR_MEMORY;

    if (!read_file(input_path, &input_bytes, &input_size))
        return MCO2_Q8_ERR_IO;
    if (input_size % sizeof(uint32_t) != 0) {
        status = MCO2_Q8_ERR_PAYLOAD_LENGTH;
        goto done;
    }
    count = input_size / sizeof(uint32_t);
    if (count > SIZE_MAX / sizeof *values ||
        count > SIZE_MAX - MCO2_Q8_HEADER_SIZE) {
        status = MCO2_Q8_ERR_COUNT;
        goto done;
    }
    payload_bytes = bits == MCO2_Q4_BITS ? count / 2 + (count & 1) : count;
    if (payload_bytes > SIZE_MAX - MCO2_Q8_HEADER_SIZE) {
        status = MCO2_Q8_ERR_COUNT;
        goto done;
    }
    record_size = MCO2_Q8_HEADER_SIZE + payload_bytes;
    values = count == 0 ? NULL : (float *)malloc(count * sizeof *values);
    words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *words);
    generated_words = count == 0 ? NULL : (uint32_t *)malloc(count * sizeof *generated_words);
    record = (uint8_t *)malloc(record_size);
    samples = (mco2_bench_sample *)calloc((size_t)reps, sizeof *samples);
    scale_partial_count = mco2_q8_scale_workspace_elements(count);
    if (scale_partial_count == SIZE_MAX ||
        scale_partial_count > SIZE_MAX / sizeof *scale_partials) {
        status = MCO2_Q8_ERR_MEMORY;
        goto done;
    }
    if (scale_partial_count != 0)
        scale_partials = (float *)malloc(scale_partial_count * sizeof *scale_partials);
    if ((count != 0 && (values == NULL || words == NULL || generated_words == NULL)) ||
        record == NULL || samples == NULL ||
        (scale_partial_count != 0 && scale_partials == NULL)) {
        status = MCO2_Q8_ERR_MEMORY;
        goto done;
    }
    for (size_t element = 0; element < count; element++) {
        uint32_t bits32 = mco2_load_u32_le(input_bytes + 4 * element);
        memcpy(&values[element], &bits32, sizeof bits32);
    }
    if (words_path != NULL) {
        if (!read_file(words_path, &word_bytes, &word_size)) {
            status = MCO2_Q8_ERR_IO;
            goto done;
        }
        if (count > SIZE_MAX / sizeof(uint32_t) ||
            word_size != count * sizeof(uint32_t)) {
            status = MCO2_Q8_ERR_PAYLOAD_LENGTH;
            goto done;
        }
        for (size_t element = 0; element < count; element++)
            words[element] = mco2_load_u32_le(word_bytes + 4 * element);
    }
    status = mco2_q8_validate_input(values, count);
    if (status != MCO2_Q8_OK)
        goto done;
    if (scale_seen && (!isfinite(prescribed_scale) || prescribed_scale < 0.0f)) {
        status = MCO2_Q8_ERR_SCALE;
        goto done;
    }

    if (strcmp(backend, "cpu") == 0) {
        status = bench_cpu_compress_one(
            (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
            scale_seen, prescribed_scale, words_path != NULL ? words : NULL,
            generated_words, scale_partials, scale_partial_count, record);
        if (status != MCO2_Q8_OK)
            goto done;
        if (output_path != NULL && !write_file(output_path, record, record_size)) {
            status = MCO2_Q8_ERR_IO;
            goto done;
        }
        if (!bench_clock_init(&clock_frequency)) {
            status = MCO2_Q8_ERR_CLOCK;
            goto done;
        }
        for (uint64_t run = 0; run < total_runs; run++) {
            double start_ms, stop_ms;
            uint64_t run_offset = run < warmups ? reps + run : run - warmups;
            uint64_t current_invocation = invocation_id + run_offset;
            if (!bench_clock_now_ms(&clock_frequency, &start_ms)) {
                status = MCO2_Q8_ERR_CLOCK;
                goto done;
            }
            status = bench_cpu_compress_one(
                (uint8_t)bits, values, count, seed, tensor_id,
                current_invocation, scale_seen, prescribed_scale,
                words_path != NULL ? words : NULL, generated_words,
                scale_partials, scale_partial_count, record);
            if (status != MCO2_Q8_OK)
                goto done;
            if (!bench_clock_now_ms(&clock_frequency, &stop_ms)) {
                status = MCO2_Q8_ERR_CLOCK;
                goto done;
            }
            if (run >= warmups)
                samples[run - warmups].wall_ms = stop_ms - start_ms;
        }
    } else {
#ifdef MCO2_ENABLE_CUDA
        uint8_t *base_payload = record + MCO2_Q8_HEADER_SIZE;
        status = mco2_cuda_bench(
            (uint8_t)bits, values, count, seed, tensor_id, invocation_id,
            scale_seen, prescribed_scale, words_path != NULL ? words : NULL,
            (int)block_size, (int)grid_size,
            strcmp(boundary_name, "resident") == 0
                ? MCO2_CUDA_BENCH_RESIDENT
                : strcmp(boundary_name, "resident-graph") == 0
                    ? MCO2_CUDA_BENCH_RESIDENT_GRAPH
                    : MCO2_CUDA_BENCH_HOST_ORIGIN,
            warmups, reps, base_payload, &scale, samples, &capture_ms);
        if (status != MCO2_Q8_OK)
            goto done;
        status = mco2_q8_header_encode((uint8_t)bits, (uint64_t)count,
                                       scale, record);
        if (status != MCO2_Q8_OK)
            goto done;
        if (output_path != NULL && !write_file(output_path, record, record_size)) {
            status = MCO2_Q8_ERR_IO;
            goto done;
        }
#endif
    }

    print_bench_json(backend, strcmp(backend, "cpu") == 0 ? "host-host" : boundary_name,
                     (uint8_t)bits, count, seed, tensor_id, invocation_id,
                     warmups, reps, scale_seen, prescribed_scale,
                     (int)block_size, (int)grid_size, payload_bytes, samples,
                     capture_ms);
    status = MCO2_Q8_OK;

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

static mco2_q8_status decompress_file(int argc, char **argv)
{
    const char *input_path = NULL, *output_path = NULL;
    uint8_t *record = NULL, *output = NULL;
    float *values = NULL;
    size_t record_size = 0, count = 0, output_size, i;
    mco2_q8_status status;
    int argument;

    for (argument = 2; argument < argc; argument++) {
        const char *option = argv[argument];
        if (argument + 1 >= argc)
            return MCO2_Q8_ERR_ARGUMENT;
        if (strcmp(option, "--input") == 0)
            input_path = argv[++argument];
        else if (strcmp(option, "--output") == 0)
            output_path = argv[++argument];
        else
            return MCO2_Q8_ERR_ARGUMENT;
    }
    if (input_path == NULL || output_path == NULL)
        return MCO2_Q8_ERR_ARGUMENT;
    if (!read_file(input_path, &record, &record_size))
        return MCO2_Q8_ERR_IO;
    status = mco2_q8_decode_record(record, record_size, &values, &count);
    if (status != MCO2_Q8_OK)
        goto done;
    if (count > SIZE_MAX / sizeof(uint32_t)) {
        status = MCO2_Q8_ERR_COUNT;
        goto done;
    }
    output_size = count * sizeof(uint32_t);
    output = output_size == 0 ? NULL : (uint8_t *)malloc(output_size);
    if (output_size != 0 && output == NULL) {
        status = MCO2_Q8_ERR_MEMORY;
        goto done;
    }
    for (i = 0; i < count; i++) {
        uint32_t bits32;
        memcpy(&bits32, &values[i], sizeof bits32);
        mco2_store_u32_le(output + 4 * i, bits32);
    }
    if (!write_file(output_path, output, output_size))
        status = MCO2_Q8_ERR_IO;

done:
    free(output);
    free(values);
    free(record);
    return status;
}

int main(int argc, char **argv)
{
    mco2_q8_status status;

    if (argc < 2 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        usage(argc < 2 ? stderr : stdout);
        return argc < 2 ? 2 : 0;
    }
    if (strcmp(argv[1], "compress") == 0)
        status = compress_file(argc, argv);
    else if (strcmp(argv[1], "bench") == 0)
        status = bench_file(argc, argv);
    else if (strcmp(argv[1], "decompress") == 0)
        status = decompress_file(argc, argv);
    else {
        usage(stderr);
        return 2;
    }
    if (status != MCO2_Q8_OK) {
        fprintf(stderr, "%s: %s\n", argv[1], mco2_q8_status_message(status));
        return 1;
    }
    return 0;
}
