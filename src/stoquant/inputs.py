"""Input families for the benchmark driver and the Layer 3 suite.

Dense inputs are standard-normal FP32 vectors. Sparse inputs are the dense
vector of the same seed and length with a Bernoulli mask of +0.0 values drawn
from a separate recorded seed. Model inputs are synthetic tensors with the
parameter shapes of ResNet-18 for CIFAR-10; they carry no trained values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_INPUT_SEED = 2026
SPARSE_MASK_SEED = 1021
SPARSE_ZERO_FRACTION = 0.9
BIAS_STD = 0.01
INPUT_FAMILIES = ("dense", "sparse", "model")
MODEL_TENSOR_SETS = ("distinct", "all")
MODEL_NAME = "resnet18-cifar10"


@dataclass(frozen=True)
class ModelTensor:
    name: str
    shape: tuple[int, ...]
    kind: str  # "conv", "linear", "bias", or "bn"

    @property
    def count(self) -> int:
        return math.prod(self.shape)

    @property
    def fan_in(self) -> int:
        return math.prod(self.shape[1:])


def _resnet18_cifar_tensors() -> tuple[ModelTensor, ...]:
    """Parameter tensors of ResNet-18 for CIFAR-10, in module order.

    The stem is a 3x3 convolution with 64 channels and no max pool; each stage
    has two basic blocks; the first block of stages 2-4 has a 1x1 projection
    shortcut; the classifier maps 512 features to 10 classes.
    """
    tensors: list[ModelTensor] = []

    def conv_bn(prefix: str, bn: str, out_ch: int, in_ch: int, k: int) -> None:
        tensors.append(ModelTensor(f"{prefix}.weight", (out_ch, in_ch, k, k), "conv"))
        tensors.append(ModelTensor(f"{bn}.weight", (out_ch,), "bn"))
        tensors.append(ModelTensor(f"{bn}.bias", (out_ch,), "bn"))

    conv_bn("conv1", "bn1", 64, 3, 3)
    in_ch = 64
    for stage, out_ch in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            name = f"layer{stage}.{block}"
            conv_bn(f"{name}.conv1", f"{name}.bn1", out_ch, in_ch, 3)
            conv_bn(f"{name}.conv2", f"{name}.bn2", out_ch, out_ch, 3)
            if block == 0 and in_ch != out_ch:
                conv_bn(f"{name}.shortcut.0", f"{name}.shortcut.1", out_ch, in_ch, 1)
            in_ch = out_ch
    tensors.append(ModelTensor("linear.weight", (10, 512), "linear"))
    tensors.append(ModelTensor("linear.bias", (10,), "bias"))
    return tuple(tensors)


RESNET18_CIFAR_TENSORS = _resnet18_cifar_tensors()


def dense_vectors(counts: Sequence[int], seed: int = DEFAULT_INPUT_SEED) -> list[np.ndarray]:
    """Standard-normal FP32 vectors, drawn in order from one generator."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=count).astype(np.float32) for count in counts]


def sparsify(
    dense: np.ndarray,
    mask_seed: int = SPARSE_MASK_SEED,
    zero_fraction: float = SPARSE_ZERO_FRACTION,
) -> tuple[np.ndarray, float]:
    """Zero each element with probability `zero_fraction`; return the realised fraction."""
    mask = np.random.default_rng(mask_seed).random(dense.size) < zero_fraction
    sparse = dense.copy()
    sparse[mask] = np.float32(0.0)
    realised = float(mask.mean()) if dense.size else 0.0
    return sparse, realised


def select_model_tensors(mode: str = "distinct", limit: int | None = None) -> list[ModelTensor]:
    """All tensors, or the first tensor of each distinct (shape, kind), in module order."""
    if mode not in MODEL_TENSOR_SETS:
        raise ValueError(f"unknown model tensor set {mode!r}; choose from {MODEL_TENSOR_SETS}")
    if limit is not None and limit < 1:
        raise ValueError("the model tensor limit must be at least 1")
    if mode == "all":
        selected = list(RESNET18_CIFAR_TENSORS)
    else:
        seen: set[tuple[tuple[int, ...], str]] = set()
        selected = []
        for tensor in RESNET18_CIFAR_TENSORS:
            if (tensor.shape, tensor.kind) not in seen:
                seen.add((tensor.shape, tensor.kind))
                selected.append(tensor)
    return selected[:limit] if limit is not None else selected


def model_tensor_by_name(name: str) -> ModelTensor:
    for tensor in RESNET18_CIFAR_TENSORS:
        if tensor.name == name:
            return tensor
    raise KeyError(name)


def model_tensor_seed(tensor: ModelTensor, seed: int = DEFAULT_INPUT_SEED) -> list[int]:
    """Generator seed of a tensor: the input seed and the tensor's module index."""
    return [seed, RESNET18_CIFAR_TENSORS.index(tensor)]


def model_tensor_std(tensor: ModelTensor) -> float:
    return math.sqrt(2.0 / tensor.fan_in) if tensor.kind in ("conv", "linear") else BIAS_STD


def model_tensor_values(tensor: ModelTensor, seed: int = DEFAULT_INPUT_SEED) -> np.ndarray:
    """Flattened FP32 values: He-normal weights, N(0, 0.01) biases and BN parameters.

    Codes depend on each value's ratio to the tensor's L2 scale, so a uniform
    rescaling changes them only through FP32 rounding edge effects.
    """
    rng = np.random.default_rng(model_tensor_seed(tensor, seed))
    return (rng.normal(size=tensor.count) * model_tensor_std(tensor)).astype(np.float32)


def course_vector() -> tuple[np.ndarray, float]:
    """The Layer 3 course-subset vector of 1,024 elements and its design scale 2.0.

    It holds signed zeros, the exact points +/-scale, fractions k/7 and k/127 of
    the scale on both signs, and a linear ramp inside +/-1.95.
    """
    n = 1024
    x = np.empty(n, dtype=np.float32)
    x[0:4] = [0.0, -0.0, 2.0, -2.0]
    idx = 4
    for k in range(1, 7):
        x[idx] = 2.0 * (k / 7.0)
        x[idx + 1] = -2.0 * (k / 7.0)
        idx += 2
    for k in range(1, 127):
        x[idx] = 2.0 * (k / 127.0)
        x[idx + 1] = -2.0 * (k / 127.0)
        idx += 2
    x[idx:] = np.linspace(-1.95, 1.95, n - idx, dtype=np.float32)
    return x, 2.0
