from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class RowSample:
    y: int
    width: int
    labels: np.ndarray
    lengths: np.ndarray
    tokens: np.ndarray
    token_types: np.ndarray
    rem_width: np.ndarray
    run_starts: np.ndarray


def encode_row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    if not isinstance(row, np.ndarray) or row.ndim != 1:
        raise ValueError("row must be a 1D numpy array")
    w = int(row.shape[0])
    if w <= 0:
        raise ValueError("row width must be > 0")

    changes = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [w]))

    runs: List[Tuple[int, int]] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        runs.append((int(row[s]), int(e - s)))
    return runs


def decode_row_runs(runs: Sequence[Tuple[int, int]], width: int, dtype: np.dtype = np.uint16) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be > 0")
    out = np.empty((width,), dtype=dtype)
    x = 0
    for label, run_len in runs:
        if run_len <= 0:
            raise ValueError("run_len must be >= 1")
        x2 = x + int(run_len)
        if x2 > width:
            raise ValueError("runs exceed target width")
        out[x:x2] = label
        x = x2
    if x != width:
        raise ValueError("runs do not fill width")
    return out


def encode_mask_to_row_tokens(mask: np.ndarray) -> List[RowSample]:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D numpy array")
    h, w = int(mask.shape[0]), int(mask.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("mask shape must be positive")

    rows: List[RowSample] = []
    for y in range(h):
        runs = encode_row_runs(mask[y])
        n_runs = len(runs)
        labels = np.array([lab for lab, _ in runs], dtype=np.int64)
        lengths = np.array([run_len for _, run_len in runs], dtype=np.int64)

        tokens = np.empty((2 * n_runs,), dtype=np.int64)
        token_types = np.empty((2 * n_runs,), dtype=np.int64)
        rem_width = np.zeros((2 * n_runs,), dtype=np.int64)
        run_starts = np.empty((n_runs,), dtype=np.int64)

        x = 0
        for j, (lab, run_len) in enumerate(runs):
            t0 = 2 * j
            t1 = t0 + 1
            tokens[t0] = lab
            tokens[t1] = run_len
            token_types[t0] = 0
            token_types[t1] = 1
            run_starts[j] = x
            rem_width[t1] = w - x
            x += run_len

        rows.append(
            RowSample(
                y=y,
                width=w,
                labels=labels,
                lengths=lengths,
                tokens=tokens,
                token_types=token_types,
                rem_width=rem_width,
                run_starts=run_starts,
            )
        )
    return rows


def decode_mask_from_row_tokens(rows: Sequence[RowSample], height: int, width: int, dtype: np.dtype = np.uint16) -> np.ndarray:
    if len(rows) != height:
        raise ValueError("row count does not match height")
    out = np.empty((height, width), dtype=dtype)
    for y, row in enumerate(rows):
        runs = list(zip(row.labels.tolist(), row.lengths.tolist()))
        out[y] = decode_row_runs(runs, width=width, dtype=dtype)
    return out
