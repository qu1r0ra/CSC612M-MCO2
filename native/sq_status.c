#include "sq_status.h"

const char *sq_status_message(sq_status status) {
  switch (status) {
  case SQ_OK:
    return "ok";
  case SQ_ERR_ARGUMENT:
    return "invalid argument";
  case SQ_ERR_MAGIC:
    return "bad record magic";
  case SQ_ERR_VERSION:
    return "unsupported record version";
  case SQ_ERR_BIT_WIDTH:
    return "unsupported bit width; this tool accepts 4-bit and 8-bit records";
  case SQ_ERR_RESERVED:
    return "reserved header bytes must be zero";
  case SQ_ERR_COUNT:
    return "element count is too large";
  case SQ_ERR_PAYLOAD_LENGTH:
    return "payload length does not match element count";
  case SQ_ERR_SCALE:
    return "scale must be finite and non-negative; "
           "empty and all-zero inputs require zero scale";
  case SQ_ERR_CODE:
    return "record contains an invalid code";
  case SQ_ERR_ZERO_SCALE_CODE:
    return "zero-scale records must contain only the center code";
  case SQ_ERR_PADDING_NIBBLE:
    return "unused padding nibble must be zero";
  case SQ_ERR_NONFINITE:
    return "input contains a non-finite FP32 value";
  case SQ_ERR_SCALE_OVERFLOW:
    return "FP32 L2 scale overflow";
  case SQ_ERR_ID_OVERFLOW:
    return "tensor and invocation identifiers must fit in uint32";
  case SQ_ERR_MEMORY:
    return "unable to allocate memory";
  case SQ_ERR_IO:
    return "file read or write failed";
  case SQ_ERR_CUDA_UNAVAILABLE:
    return "CUDA backend is unavailable: this executable was built without "
           "CUDA support";
  case SQ_ERR_CUDA:
    return "CUDA device unavailable or CUDA runtime operation failed";
  case SQ_ERR_TIMINGS_BACKEND:
    return "--timings requires --backend cuda";
  case SQ_ERR_CLOCK:
    return "high-resolution monotonic clock failed";
  case SQ_ERR_AVX2_UNAVAILABLE:
    return "cpu-avx2 backend needs a processor and operating system with AVX2 "
           "enabled";
  }
  return "unknown error";
}
