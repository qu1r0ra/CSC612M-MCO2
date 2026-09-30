import numpy as np
import pytest
from _bench_test_support import READY_FACTS, ROOT

from stoquant.inputs import (
    RESNET18_CIFAR_TENSORS,
    SPARSE_MASK_SEED,
    generate_family_inputs,
    generate_inputs,
    select_model_tensors,
)
from stoquant.matrix import run_benchmark_matrix


def test_dense_family_inputs_match_revision_3_generator(tmp_path):
    legacy = generate_inputs([1024, 4096], tmp_path / "legacy")
    family = generate_family_inputs("dense", [1024, 4096], tmp_path / "family")
    assert list(family) == ["n1024", "n4096"]
    for count in (1024, 4096):
        meta = family[f"n{count}"]
        assert meta.provenance.get("input_family") == "dense"
        assert meta.provenance.get("sha256") == legacy[count].provenance.get("sha256")
        assert meta.path.read_bytes() == legacy[count].path.read_bytes()


def test_sparse_family_masks_the_dense_input(tmp_path):
    dense = generate_family_inputs("dense", [1024, 16384], tmp_path / "dense")
    sparse = generate_family_inputs("sparse", [1024, 16384], tmp_path / "sparse")
    assert list(sparse) == ["sparse_n1024", "sparse_n16384"]
    for count in (1024, 16384):
        meta = sparse[f"sparse_n{count}"]
        values = np.frombuffer(meta.path.read_bytes(), dtype="<f4")
        source = np.frombuffer(dense[f"n{count}"].path.read_bytes(), dtype="<f4")
        zeros = values == 0
        assert not np.signbit(values[zeros]).any()
        np.testing.assert_array_equal(values[~zeros], source[~zeros])
        assert meta.provenance.get("zero_fraction_realised") == pytest.approx(zeros.mean())
        assert abs(meta.provenance.get("zero_fraction_realised", 0.0) - 0.9) < 0.03
        assert meta.provenance.get("mask_seed") == SPARSE_MASK_SEED
        assert meta.provenance["count"] == count


def test_model_family_records_tensor_provenance(tmp_path):
    inputs = generate_family_inputs("model", [], tmp_path, model_limit=3)
    assert list(inputs) == [
        "model_conv1.weight",
        "model_bn1.weight",
        "model_layer1.0.conv1.weight",
    ]
    conv = inputs["model_conv1.weight"]
    assert conv.provenance.get("tensor_shape") == [64, 3, 3, 3]
    assert conv.provenance["count"] == 1728
    assert conv.provenance.get("seed") == [2026, 0]
    assert conv.provenance.get("std") == pytest.approx(np.sqrt(2.0 / 27))
    assert conv.path.stat().st_size == 4 * 1728
    assert inputs["model_bn1.weight"].provenance.get("std") == 0.01


def test_resnet18_cifar_tensor_table():
    assert len(RESNET18_CIFAR_TENSORS) == 62
    assert sum(t.count for t in RESNET18_CIFAR_TENSORS) == 11_173_962
    assert len(select_model_tensors("distinct")) == 17
    assert len(select_model_tensors("all", 5)) == 5
    with pytest.raises(ValueError):
        select_model_tensors("some")
    with pytest.raises(ValueError):
        select_model_tensors("all", 0)


def test_driver_rejects_model_options_outside_model_family(tmp_path):
    with pytest.raises(ValueError, match="model family"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=tmp_path / "out",
            counts=[1024],
            bit_widths=[8],
            backends=["cpu"],
            input_family="sparse",
            model_limit=2,
            readiness_facts=READY_FACTS,
            allow_dirty=True,
        )
    with pytest.raises(ValueError, match="model family"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=tmp_path / "out",
            counts=[1024],
            bit_widths=[8],
            backends=["cpu"],
            input_family="dense",
            model_tensors="distinct",
            readiness_facts=READY_FACTS,
            allow_dirty=True,
        )
    with pytest.raises(ValueError, match="unknown input family"):
        run_benchmark_matrix(
            root=ROOT,
            output_dir=tmp_path / "out",
            counts=[1024],
            bit_widths=[8],
            backends=["cpu"],
            input_family="images",
            readiness_facts=READY_FACTS,
            allow_dirty=True,
        )
    assert not (tmp_path / "out").exists()
