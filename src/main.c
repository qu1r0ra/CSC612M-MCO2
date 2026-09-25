#include "codec.h"
#include "byteorder.h"
#include "quantizer.h"
#include "rng_cpu.h"
#ifdef MCO2_ENABLE_CUDA
#include "quantizer_cuda.h"
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *stream)
{
    fprintf(stream,
            "Usage:\n"
            "  mco2 compress --input INPUT.f32 --output RECORD --seed UINT64\n"
            "      [--backend cpu|cuda] [--bits 4|8] [--tensor-id UINT32]\n"
            "      [--invocation-id UINT32] [--scale FP32] [--words WORDS.u32]\n"
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
