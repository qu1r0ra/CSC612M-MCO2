#ifndef SQ_CUDA_LIMITS_H
#define SQ_CUDA_LIMITS_H

/* Launch limits and K1 variants the CLI validates against. Plain constants, so
   a CPU-only build reads them without the CUDA header. */

#define SQ_CUDA_MAX_BLOCK_SIZE 1024
#define SQ_CUDA_MAX_GRID_SIZE 65535

/* K1 variants (issue #23). Both write the same scale bit for bit. */
enum { SQ_CUDA_K1_REFERENCE = 0, SQ_CUDA_K1_OPTIMIZED = 1 };

#endif
