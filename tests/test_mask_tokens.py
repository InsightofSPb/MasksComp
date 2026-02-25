import numpy as np
import pytest

from tools.mask_tokens import decode_mask_rle_scanline, encode_mask_rle_scanline


def test_round_trip_random_masks():
    rng = np.random.default_rng(12345)
    for _ in range(20):
        h = int(rng.integers(1, 129))
        w = int(rng.integers(1, 129))
        mask = rng.integers(0, 2, size=(h, w), dtype=np.uint8)

        tokens = encode_mask_rle_scanline(mask)
        decoded = decode_mask_rle_scanline(tokens)

        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded, mask)


def test_round_trip_block_masks():
    cases = []

    m1 = np.zeros((8, 10), dtype=np.uint8)
    m1[2:6, 3:8] = 1
    cases.append(m1)

    m2 = np.zeros((17, 13), dtype=np.uint8)
    m2[0:5, 0:4] = 1
    m2[7:12, 5:11] = 1
    cases.append(m2)

    m3 = np.zeros((32, 32), dtype=np.uint8)
    m3[4:28, 4:28] = 1
    m3[10:22, 10:22] = 0
    cases.append(m3)

    m4 = np.zeros((15, 40), dtype=np.uint8)
    m4[:, 5:7] = 1
    m4[6:10, 20:39] = 1
    cases.append(m4)

    m5 = np.zeros((9, 9), dtype=np.uint8)
    m5[1:8, 1:8] = 1
    m5[2:7, 2:7] = 0
    m5[3:6, 3:6] = 1
    cases.append(m5)

    for mask in cases:
        tokens = encode_mask_rle_scanline(mask)
        decoded = decode_mask_rle_scanline(tokens)
        assert np.array_equal(decoded, mask)


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([999, 2, 2, 0, 4, -2], dtype=np.int32),  # wrong BOS
        np.array([-1, 2, 2, 0, 4], dtype=np.int32),  # missing EOS
        np.array([-1, 2, 2, 0, 3, -2], dtype=np.int32),  # sum(run_len) != H*W
    ],
)
def test_decode_broken_tokens_raise_value_error(tokens):
    with pytest.raises(ValueError):
        decode_mask_rle_scanline(tokens)
