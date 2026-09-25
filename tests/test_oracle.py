import numpy as np
import pytest

from mco2_oracle import (
    compress_record_fp32,
    decode_record,
    philox4x32_10,
    philox_block,
    philox_words,
    scale_fp32,
)


@pytest.mark.parametrize(
    ("counter", "key", "expected"),
    [
        (
            (0, 0, 0, 0),
            (0, 0),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        ),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_matches_random123_known_answers(counter, key, expected):
    counters = np.asarray([counter], dtype=np.uint32)
    keys = np.asarray(key, dtype=np.uint32)

    actual = philox4x32_10(counters, keys)

    assert tuple(int(word) for word in actual[0]) == expected


def test_seed_group_tensor_invocation_mapping_matches_contract():
    seed = 0x0123456789ABCDEF
    key = np.asarray((0x89ABCDEF, 0x01234567), dtype=np.uint32)
    words = philox_words(16, seed, tensor_id=7, invocation_id=9)

    for group in range(4):
        counter = np.asarray([(group, 0, 7, 9)], dtype=np.uint32)
        expected = philox4x32_10(counter, key)[0]
        np.testing.assert_array_equal(words[group * 4 : group * 4 + 4], expected)

    assert int(words[13]) == int(
        philox4x32_10(np.asarray([(3, 0, 7, 9)], dtype=np.uint32), key)[0, 1]
    )


def test_philox_splits_a_64_bit_group_into_low_and_high_words():
    seed = 0x0123456789ABCDEF
    key = np.asarray((0x89ABCDEF, 0x01234567), dtype=np.uint32)
    expected = philox4x32_10(np.asarray([(2, 1, 7, 9)], dtype=np.uint32), key)[0]

    actual = philox_block(seed, 0x100000002, tensor_id=7, invocation_id=9)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tensor_id,invocation_id", [(1 << 32, 0), (0, 1 << 32)])
def test_philox_rejects_identifier_overflow(tensor_id, invocation_id):
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        philox_block(1, 0, tensor_id=tensor_id, invocation_id=invocation_id)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], 0.0), ([0.0, -0.0, 0.0], 0.0), ([3.0, 4.0], 5.0)],
)
def test_fp32_scale_handles_empty_zero_and_known_norm(values, expected):
    assert float(scale_fp32(values)) == expected


def test_fp32_oracle_serializes_prescribed_word_record():
    values = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
    words = np.asarray([0, 0, 0, 0xFFFFFFFF, 0], dtype=np.uint32)
    expected = bytes.fromhex(
        "4d 53 51 31 01 08 00 00 05 00 00 00 00 00 00 00 00 00 00 40 00 3f 7f be fe"
    )

    actual = compress_record_fp32(values, scale=2.0, words=words)

    assert actual == expected


def test_fp32_oracle_serializes_and_decodes_4bit_prescribed_word_record():
    values = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
    words = np.asarray([0, 0, 0, 0xFFFFFFFF, 0], dtype=np.uint32)
    expected = bytes.fromhex("4d 53 51 31 01 04 00 00 05 00 00 00 00 00 00 00 00 00 00 40 30 a7 0e")

    actual = compress_record_fp32(values, bits=4, scale=2.0, words=words)
    assert actual == expected

    decoded = decode_record(actual)
    assert len(decoded) == 5
    assert decoded[0] == -2.0
    assert decoded[2] == 0.0
    assert decoded[4] == 2.0
