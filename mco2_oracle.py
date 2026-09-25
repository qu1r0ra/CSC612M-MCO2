"""Independent NumPy oracle for the course CPU quantization pipeline."""

from __future__ import annotations

import math
import operator
import struct

import numpy as np

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1
_PHILOX_M0 = np.uint64(0xD2511F53)
_PHILOX_M1 = np.uint64(0xCD9E8D57)
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_BLOCK_SIZE = 256
_BITS = 8
_SIGNED_LIMIT = 127
_HEADER = struct.Struct("<4sBBHQf")
_MAGIC = b"MSQ1"
_VERSION = 1


def _unsigned(value: int, bits: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0 or result >= 1 << bits:
        raise ValueError(f"{name} must fit in an unsigned {bits}-bit integer")
    return result


def _word_array(values: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    limit = 1 << width
    if np.any(array < 0) or np.any(array >= limit):
        raise ValueError(f"{name} contains a value outside uint{width}")
    return array.astype(np.uint32, copy=False)


def philox4x32_10(counters: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Apply Random123-compatible Philox4x32-10 to one or more counters."""
    counter_words = _word_array(counters, 32, "counters")
    if counter_words.ndim < 1 or counter_words.shape[-1] != 4:
        raise ValueError("counters must have a final dimension of four words")
    key_words = _word_array(key, 32, "key")
    if key_words.shape != (2,):
        raise ValueError("key must contain two words")

    c0, c1, c2, c3 = (counter_words[..., index].astype(np.uint64) for index in range(4))
    k0, k1 = (int(word) for word in key_words)

    for round_index in range(10):
        product0 = _PHILOX_M0 * c0
        product1 = _PHILOX_M1 * c2
        low0 = product0 & np.uint64(_MASK32)
        high0 = product0 >> np.uint64(32)
        low1 = product1 & np.uint64(_MASK32)
        high1 = product1 >> np.uint64(32)
        c0, c1, c2, c3 = (
            high1 ^ c1 ^ np.uint64(k0),
            low1,
            high0 ^ c3 ^ np.uint64(k1),
            low0,
        )
        if round_index != 9:
            k0 = (k0 + _PHILOX_W0) & _MASK32
            k1 = (k1 + _PHILOX_W1) & _MASK32

    return np.stack((c0, c1, c2, c3), axis=-1).astype(np.uint32)


def philox_block(seed: int, group: int, tensor_id: int, invocation_id: int) -> np.ndarray:
    """Return the four words for one (seed, group, tensor, invocation) counter."""
    seed = _unsigned(seed, 64, "seed")
    group = _unsigned(group, 64, "group")
    tensor_id = _unsigned(tensor_id, 32, "tensor_id")
    invocation_id = _unsigned(invocation_id, 32, "invocation_id")
    key = np.array([seed & _MASK32, seed >> 32], dtype=np.uint32)
    counter = np.array([group & _MASK32, group >> 32, tensor_id, invocation_id], dtype=np.uint32)
    return philox4x32_10(counter, key)


def philox_words(count: int, seed: int, tensor_id: int = 0, invocation_id: int = 0) -> np.ndarray:
    """Generate one Philox word per element using the contract's group/lane map."""
    count = operator.index(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    seed = _unsigned(seed, 64, "seed")
    tensor_id = _unsigned(tensor_id, 32, "tensor_id")
    invocation_id = _unsigned(invocation_id, 32, "invocation_id")

    group_count = (count + 3) // 4
    groups = np.arange(group_count, dtype=np.uint64)
    counters = np.empty((group_count, 4), dtype=np.uint32)
    counters[:, 0] = groups.astype(np.uint32)
    counters[:, 1] = (groups >> np.uint64(32)).astype(np.uint32)
    counters[:, 2] = tensor_id
    counters[:, 3] = invocation_id
    key = np.array([seed & _MASK32, seed >> 32], dtype=np.uint32)
    return philox4x32_10(counters, key).reshape(-1)[:count].copy()


def _values_fp32(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("input contains a non-finite FP32 value")
    return array


def _pairwise_sum_fp32(values: np.ndarray) -> np.float32:
    count = len(values)
    padded_count = 1 << max(0, (count - 1).bit_length())
    work = np.zeros(padded_count, dtype=np.float32)
    work[:count] = values
    while len(work) > 1:
        work = np.add(work[0::2], work[1::2])
    return np.float32(work[0])


def scale_fp32(values: np.ndarray) -> np.float32:
    """Compute a max-rescaled L2 scale with the contract's fixed pairwise order."""
    array = _values_fp32(values)
    count = len(array)
    if count == 0:
        return np.float32(0)

    max_abs = np.max(np.abs(array), initial=np.float32(0))
    if max_abs == 0:
        return np.float32(0)

    partials = []
    abs_values = np.abs(array)
    for start in range(0, count, _BLOCK_SIZE):
        block = abs_values[start : start + _BLOCK_SIZE]
        ratios = np.divide(block, max_abs)
        squares = np.multiply(ratios, ratios)
        terms = np.zeros(_BLOCK_SIZE, dtype=np.float32)
        terms[: len(block)] = squares
        partials.append(_pairwise_sum_fp32(terms))

    total = _pairwise_sum_fp32(np.asarray(partials, dtype=np.float32))
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.float32(max_abs * np.sqrt(total))
    if not np.isfinite(scale):
        raise OverflowError("FP32 L2 scale overflow")
    return scale


def scale_fp64(values: np.ndarray) -> float:
    """Compute the FP64 reference norm for an FP32 input vector."""
    array = _values_fp32(values).astype(np.float64)
    if len(array) == 0:
        return 0.0
    max_abs = float(np.max(np.abs(array), initial=0.0))
    if max_abs == 0:
        return 0.0
    ratios = np.abs(array) / max_abs
    return max_abs * math.sqrt(float(np.sum(ratios * ratios, dtype=np.float64)))


def _resolve_words(
    count: int,
    *,
    seed: int,
    tensor_id: int,
    invocation_id: int,
    words: np.ndarray | None,
) -> np.ndarray:
    if words is None:
        return philox_words(count, seed, tensor_id, invocation_id)
    array = _word_array(words, 32, "words")
    if array.shape != (count,):
        raise ValueError("prescribed random words must match the input element count")
    return array


def _resolve_scale_fp32(array: np.ndarray, scale: float | None) -> np.float32:
    if scale is None:
        return scale_fp32(array)
    try:
        value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("prescribed scale must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("prescribed scale must be a finite non-negative number")
    with np.errstate(over="ignore", under="ignore"):
        scale32 = np.float32(value)
    if not np.isfinite(scale32):
        raise OverflowError("prescribed scale is outside the FP32 range")
    if not np.any(array) and scale32 != 0:
        raise ValueError("empty and all-zero inputs require zero scale")
    if scale32 == 0 and np.any(array != 0):
        raise ValueError("zero scale cannot represent a nonzero input")
    return scale32


def _round_codes_fp32(array: np.ndarray, scale: np.float32, words: np.ndarray) -> np.ndarray:
    if scale == 0:
        return np.full(len(array), _SIGNED_LIMIT, dtype=np.uint8)
    ratios = np.divide(np.abs(array), scale)
    scaled = np.multiply(ratios, np.float32(_SIGNED_LIMIT))
    scaled = np.minimum(scaled, np.float32(_SIGNED_LIMIT))
    lower = np.floor(scaled).astype(np.int32)
    fractions = np.subtract(scaled, lower.astype(np.float32))
    thresholds = np.floor(fractions.astype(np.float64) * (1 << 32)).astype(np.uint64)
    rounded = words.astype(np.uint64) < thresholds
    magnitude = lower + rounded.astype(np.int32)
    signed = np.where(np.signbit(array) & (magnitude != 0), -magnitude, magnitude)
    return (signed + _SIGNED_LIMIT).astype(np.uint8)


def compress_record_fp32(
    values: np.ndarray,
    *,
    seed: int = 0,
    tensor_id: int = 0,
    invocation_id: int = 0,
    scale: float | None = None,
    words: np.ndarray | None = None,
) -> bytes:
    """Return an 8-bit record, using either the mapped stream or prescribed words."""
    array = _values_fp32(values)
    scale32 = _resolve_scale_fp32(array, scale)
    random_words = _resolve_words(
        len(array), seed=seed, tensor_id=tensor_id, invocation_id=invocation_id, words=words
    )
    codes = _round_codes_fp32(array, scale32, random_words)
    header = _HEADER.pack(_MAGIC, _VERSION, _BITS, 0, len(array), float(scale32))
    return header + codes.tobytes()


def reference_fp64(
    values: np.ndarray,
    *,
    seed: int = 0,
    tensor_id: int = 0,
    invocation_id: int = 0,
    words: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the FP64 scale and reconstruction for the same logical RNG stream."""
    array = _values_fp32(values).astype(np.float64)
    scale = scale_fp64(array.astype(np.float32))
    random_words = _resolve_words(
        len(array), seed=seed, tensor_id=tensor_id, invocation_id=invocation_id, words=words
    )
    if scale == 0:
        return scale, np.zeros(len(array), dtype=np.float64)

    scaled = np.minimum(np.abs(array) / scale * _SIGNED_LIMIT, _SIGNED_LIMIT)
    lower = np.floor(scaled)
    fractions = scaled - lower
    thresholds = np.floor(fractions * (1 << 32)).astype(np.uint64)
    rounded = random_words.astype(np.uint64) < thresholds
    magnitude = lower.astype(np.int64) + rounded.astype(np.int64)
    signed = np.where(np.signbit(array) & (magnitude != 0), -magnitude, magnitude)
    decoded = (signed.astype(np.float64) / _SIGNED_LIMIT) * scale
    return scale, decoded


def decode_record(record: bytes | bytearray | memoryview) -> np.ndarray:
    """Validate and decode a course-minimum 8-bit record."""
    data = bytes(record)
    if len(data) < _HEADER.size:
        raise ValueError("record is shorter than the 20-byte header")
    magic, version, bits, reserved, count, scale = _HEADER.unpack_from(data)
    if magic != _MAGIC:
        raise ValueError("bad record magic")
    if version != _VERSION:
        raise ValueError("unsupported record version")
    if bits != _BITS:
        raise ValueError("unsupported bit width; this tool accepts 8-bit records")
    if reserved != 0:
        raise ValueError("reserved header bytes must be zero")
    if count != len(data) - _HEADER.size:
        raise ValueError("payload length does not match element count")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")

    codes = np.frombuffer(data, dtype=np.uint8, count=count, offset=_HEADER.size)
    if np.any(codes > 2 * _SIGNED_LIMIT):
        raise ValueError("record contains an invalid 8-bit code")
    if scale == 0:
        if np.any(codes != _SIGNED_LIMIT):
            raise ValueError("zero-scale records must contain only the center code")
        return np.zeros(count, dtype=np.float32)

    signed = codes.astype(np.int16) - _SIGNED_LIMIT
    fractions = np.divide(signed.astype(np.float32), np.float32(_SIGNED_LIMIT))
    return np.multiply(fractions, np.float32(scale))
