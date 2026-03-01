from __future__ import annotations

import numpy as np
import pytest

from maskscomp.datasets.a2d2 import rgb_mask_to_id


def test_rgb_mask_to_id_basic_conversion_uint8():
    rgb2id = {
        (255, 0, 0): 0,
        (0, 255, 0): 1,
        (0, 0, 255): 2,
    }
    mask = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[0, 255, 0], [255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    out = rgb_mask_to_id(mask, rgb2id)

    expected = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.uint8)
    assert out.dtype == np.uint8
    assert np.array_equal(out, expected)


def test_rgb_mask_to_id_unknown_color_raises():
    rgb2id = {
        (255, 0, 0): 0,
        (0, 255, 0): 1,
    }
    mask = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [123, 45, 67]],
            [[0, 255, 0], [255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="unknown RGB"):
        rgb_mask_to_id(mask, rgb2id)
