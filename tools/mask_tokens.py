"""Mask token serialization utilities (contract v0).

Format:
- Tokens dtype: int32
- BOS = -1, EOS = -2
- tokens = [BOS, H, W] + [value, run_len, value, run_len, ...] + [EOS]
- RLE is produced from mask flattened in C-order (scanline).
"""

from __future__ import annotations

import numpy as np

BOS = -1
EOS = -2


def _to_mask01(mask: np.ndarray) -> np.ndarray:
    """Normalize arbitrary mask to uint8 values 0/1."""
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got ndim={mask.ndim}")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError("mask height and width must be > 0")
    return (mask > 0).astype(np.uint8)


def encode_mask_rle_scanline(mask: np.ndarray) -> np.ndarray:
    """Encode a 2D mask into contract-v0 int32 scanline RLE tokens.

    The mask is normalized via ``(mask > 0).astype(uint8)`` before encoding.
    """
    mask01 = _to_mask01(mask)
    h, w = mask01.shape
    flat = mask01.ravel(order="C")

    payload: list[int] = []
    current_value = int(flat[0])
    run_len = 1

    for v in flat[1:]:
        v_int = int(v)
        if v_int == current_value:
            run_len += 1
        else:
            payload.extend([current_value, run_len])
            current_value = v_int
            run_len = 1

    payload.extend([current_value, run_len])

    tokens = np.array([BOS, int(h), int(w), *payload, EOS], dtype=np.int32)
    return tokens


def decode_mask_rle_scanline(tokens: np.ndarray) -> np.ndarray:
    """Decode contract-v0 int32 scanline RLE tokens back into uint8 mask 0/1."""
    if not isinstance(tokens, np.ndarray):
        raise ValueError("tokens must be a numpy.ndarray")
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be 1D, got ndim={tokens.ndim}")
    if tokens.size < 6:
        raise ValueError("tokens are too short: expected at least 6 values")

    t = tokens.astype(np.int64, copy=False)

    if int(t[0]) != BOS:
        raise ValueError(f"invalid BOS: expected {BOS}, got {int(t[0])}")
    if int(t[-1]) != EOS:
        raise ValueError(f"invalid EOS: expected {EOS}, got {int(t[-1])}")

    h = int(t[1])
    w = int(t[2])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid shape: H and W must be > 0, got H={h}, W={w}")

    payload = t[3:-1]
    if payload.size == 0:
        raise ValueError("missing RLE payload")
    if payload.size % 2 != 0:
        raise ValueError("invalid RLE payload: expected [value, run_len] pairs")

    values = payload[0::2]
    run_lens = payload[1::2]

    if np.any((values != 0) & (values != 1)):
        bad = int(values[(values != 0) & (values != 1)][0])
        raise ValueError(f"invalid RLE value: expected 0/1, got {bad}")
    if np.any(run_lens <= 0):
        bad = int(run_lens[run_lens <= 0][0])
        raise ValueError(f"invalid run_len: must be >= 1, got {bad}")

    total = int(run_lens.sum())
    expected = h * w
    if total != expected:
        raise ValueError(f"invalid RLE length: sum(run_len)={total}, expected {expected}")

    flat = np.repeat(values.astype(np.uint8), run_lens.astype(np.int64))
    return flat.reshape((h, w), order="C")
