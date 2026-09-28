#include <string.h>

#include <cuda_runtime.h>

#include "rng_cuda.h"

#define CHECK(call)                          \
    do {                                     \
        cudaError_t err_ = (call);           \
        if (err_ != cudaSuccess) {           \
            status = (int)err_;              \
            goto done;                       \
        }                                    \
    } while (0)

/* Kernels are grid-stride loops, so capping the grid never skips elements. */
#define MAX_AUTO_GRID 65535

static int auto_grid(uint64_t n, int block_size)
{
    uint64_t blocks = (n + block_size - 1) / block_size;

    return blocks < MAX_AUTO_GRID ? (int)blocks : MAX_AUTO_GRID;
}

__global__ void philox_raw_kernel(philox4x32_ctr_t ctr, philox4x32_key_t key,
                                  philox4x32_ctr_t *out)
{
    *out = philox4x32_R(MCO2_PHILOX_ROUNDS, ctr, key);
}

/* Grid-stride loop: the word for element i depends only on i, never on geometry. */
__global__ void rng_words_kernel(mco2_rng_stream s, uint64_t n, uint32_t *out)
{
    philox4x32_key_t key = mco2_philox_key(&s);
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t i;

    for (i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        philox4x32_ctr_t r = philox4x32_R(MCO2_PHILOX_ROUNDS,
                                          mco2_philox_ctr(&s, i / 4), key);
        out[i] = r.v[i % 4];
    }
}

__global__ void bernoulli_kernel(const uint32_t *words, const float *p, uint64_t n,
                                 uint8_t *out)
{
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t i;

    for (i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = (uint8_t)mco2_bernoulli(words[i], p[i]);
}

extern "C" int mco2_cuda_device_info(char *name, int name_len, int *major, int *minor)
{
    cudaDeviceProp prop;
    int status = 0;

    CHECK(cudaGetDeviceProperties(&prop, 0));
    strncpy(name, prop.name, name_len - 1);
    name[name_len - 1] = '\0';
    *major = prop.major;
    *minor = prop.minor;
done:
    return status;
}

extern "C" int mco2_philox_raw_cuda(philox4x32_ctr_t ctr, philox4x32_key_t key,
                                    philox4x32_ctr_t *out)
{
    philox4x32_ctr_t *d_out = NULL;
    int status = 0;

    CHECK(cudaMalloc(&d_out, sizeof *d_out));
    philox_raw_kernel<<<1, 1>>>(ctr, key, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out, d_out, sizeof *out, cudaMemcpyDeviceToHost));
done:
    cudaFree(d_out);
    return status;
}

extern "C" int mco2_rng_words_cuda(const mco2_rng_stream *s, uint64_t n, uint32_t *out,
                                   int block_size, int grid_size)
{
    uint32_t *d_out = NULL;
    int status = 0;

    if (block_size <= 0)
        return (int)cudaErrorInvalidValue;
    if (n == 0)
        return 0;
    if (grid_size <= 0)
        grid_size = auto_grid(n, block_size);

    CHECK(cudaMalloc(&d_out, n * sizeof *d_out));
    rng_words_kernel<<<grid_size, block_size>>>(*s, n, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out, d_out, n * sizeof *out, cudaMemcpyDeviceToHost));
done:
    cudaFree(d_out);
    return status;
}

extern "C" int mco2_bernoulli_cuda(const uint32_t *words, const float *p, uint64_t n,
                                   uint8_t *out)
{
    uint32_t *d_words = NULL;
    float *d_p = NULL;
    uint8_t *d_out = NULL;
    int block = 256;
    int status = 0;

    if (n == 0)
        return 0;

    CHECK(cudaMalloc(&d_words, n * sizeof *d_words));
    CHECK(cudaMalloc(&d_p, n * sizeof *d_p));
    CHECK(cudaMalloc(&d_out, n * sizeof *d_out));
    CHECK(cudaMemcpy(d_words, words, n * sizeof *words, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_p, p, n * sizeof *p, cudaMemcpyHostToDevice));
    bernoulli_kernel<<<auto_grid(n, block), block>>>(d_words, d_p, n, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out, d_out, n * sizeof *out, cudaMemcpyDeviceToHost));
done:
    cudaFree(d_words);
    cudaFree(d_p);
    cudaFree(d_out);
    return status;
}
