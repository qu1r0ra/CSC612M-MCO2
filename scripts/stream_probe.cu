// Device-attribute and streaming-read probe for the K1 bandwidth baseline (issue #23).
//
// Reads a float32 buffer once with a grid-stride float4 kernel and reports the
// best time over repetitions (the STREAM convention) at each power-of-two size.
// Prints one JSON object on stdout.
//
//     build/stream_probe.exe [--min-exp 10] [--max-exp 26] [--reps 50] [--discard 10]

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t check_error = (call);                                       \
        if (check_error != cudaSuccess) {                                       \
            fprintf(stderr, "%s failed: %s\n", #call,                           \
                    cudaGetErrorString(check_error));                           \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ static void read_kernel(const float4 *values, size_t count4, float *sink)
{
    float acc = 0.0f;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count4; i += stride) {
        const float4 v = values[i];
        acc += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }
    // Never true for a zeroed buffer; keeps the loads live.
    if (acc == 12345.0f)
        *sink = acc;
}

static int parse_int(int argc, char **argv, int *i, int *out)
{
    if (*i + 1 >= argc)
        return 0;
    *out = atoi(argv[++*i]);
    return 1;
}

int main(int argc, char **argv)
{
    int min_exp = 10, max_exp = 26, reps = 50, discard = 10;
    for (int i = 1; i < argc; i++) {
        int ok = 0;
        if (strcmp(argv[i], "--min-exp") == 0)
            ok = parse_int(argc, argv, &i, &min_exp);
        else if (strcmp(argv[i], "--max-exp") == 0)
            ok = parse_int(argc, argv, &i, &max_exp);
        else if (strcmp(argv[i], "--reps") == 0)
            ok = parse_int(argc, argv, &i, &reps);
        else if (strcmp(argv[i], "--discard") == 0)
            ok = parse_int(argc, argv, &i, &discard);
        if (!ok) {
            fprintf(stderr, "usage: stream_probe [--min-exp N] [--max-exp N] [--reps N] [--discard N]\n");
            return 2;
        }
    }
    if (min_exp < 2 || max_exp > 30 || min_exp > max_exp || discard < 0 || reps <= discard) {
        fprintf(stderr, "invalid probe range\n");
        return 2;
    }

    int device = 0, clock_khz = 0, bus_bits = 0, l2_bytes = 0, sms = 0, major = 0, minor = 0;
    int runtime_version = 0, driver_version = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaGetDeviceProperties(&prop, device));
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device));
    CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device));
    CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    CHECK(cudaRuntimeGetVersion(&runtime_version));
    CHECK(cudaDriverGetVersion(&driver_version));

    const size_t max_count = (size_t)1 << max_exp;
    float *values = NULL, *sink = NULL;
    CHECK(cudaMalloc(&values, max_count * sizeof(float)));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(values, 0, max_count * sizeof(float)));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    const int multipliers[] = {4, 8, 16, 32};

    printf("{\"device\":{\"name\":\"%s\",\"compute_capability\":\"%d.%d\","
           "\"memory_clock_khz\":%d,\"bus_width_bits\":%d,\"l2_bytes\":%d,"
           "\"multiprocessors\":%d,\"cuda_runtime_version\":%d,\"cuda_driver_version\":%d},"
           "\"method\":{\"kernel\":\"grid-stride float4 read, 256 threads per block\","
           "\"grids_per_multiprocessor\":[4,8,16,32],\"reps\":%d,\"discarded\":%d,"
           "\"statistic\":\"best time over kept reps and grids\"},\"sizes\":[",
           prop.name, major, minor, clock_khz, bus_bits, l2_bytes, sms, runtime_version,
           driver_version, reps, discard);
    for (int e = min_exp; e <= max_exp; e++) {
        const size_t count = (size_t)1 << e;
        float best_ms = 1e30f;
        int best_grid = 0;
        float best_grid_median_ms = 0.0f;
        for (int m : multipliers) {
            const int grid = sms * m;
            std::vector<float> kept;
            for (int r = 0; r < reps; r++) {
                CHECK(cudaEventRecord(start));
                read_kernel<<<grid, 256>>>((const float4 *)values, count / 4, sink);
                CHECK(cudaGetLastError());
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CHECK(cudaEventElapsedTime(&ms, start, stop));
                if (r >= discard)
                    kept.push_back(ms);
            }
            std::sort(kept.begin(), kept.end());
            if (kept.front() < best_ms) {
                best_ms = kept.front();
                best_grid = grid;
                const size_t half = kept.size() / 2;
                best_grid_median_ms = kept.size() % 2 ? kept[half] : 0.5f * (kept[half - 1] + kept[half]);
            }
        }
        const double bytes = (double)count * sizeof(float);
        printf("%s{\"count\":%zu,\"bytes\":%.0f,\"best_ms\":%.6f,\"best_gbps\":%.3f,"
               "\"best_grid\":%d,\"best_grid_median_ms\":%.6f}",
               e == min_exp ? "" : ",", count, bytes, best_ms, bytes / best_ms / 1e6, best_grid,
               best_grid_median_ms);
    }
    printf("]}\n");
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(values));
    CHECK(cudaFree(sink));
    return 0;
}
