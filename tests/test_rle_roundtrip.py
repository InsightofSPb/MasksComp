import numpy as np

from maskscomp.rle_tokenizer import decode_mask_from_row_tokens, encode_mask_to_row_tokens


def test_rle_roundtrip_random_multiclass_masks():
    rng = np.random.default_rng(123)
    for _ in range(20):
        h = int(rng.integers(1, 16))
        w = int(rng.integers(1, 16))
        c = int(rng.integers(2, 8))
        mask = rng.integers(0, c, size=(h, w), dtype=np.uint16)
        rows = encode_mask_to_row_tokens(mask)
        decoded = decode_mask_from_row_tokens(rows, height=h, width=w, dtype=mask.dtype)
        assert np.array_equal(decoded, mask)
