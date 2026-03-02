from __future__ import annotations

import csv
import logging
import lzma
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    sample_id: str
    prev_path: str
    cur_path: str
    t_prev: str
    t_cur: str
    split: str


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_deterministic_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def read_split_ids(splits_dir: Path, split: str) -> List[str]:
    p = Path(splits_dir) / f"facade_{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_pairs_csv(path: Path) -> List[PairRecord]:
    rows: List[PairRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        required = {"pair_id", "sample_id", "prev_path", "cur_path", "t_prev", "t_cur", "split"}
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        for r in rd:
            rows.append(
                PairRecord(
                    pair_id=str(r["pair_id"]),
                    sample_id=str(r["sample_id"]),
                    prev_path=str(r["prev_path"]),
                    cur_path=str(r["cur_path"]),
                    t_prev=str(r["t_prev"]),
                    t_cur=str(r["t_cur"]),
                    split=str(r["split"]),
                )
            )
    return rows


def write_pairs_csv(path: Path, rows: Sequence[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["pair_id", "sample_id", "prev_path", "cur_path", "t_prev", "t_cur", "split"],
        )
        wr.writeheader()
        for r in rows:
            wr.writerow(r.__dict__)


def read_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if arr.ndim == 3:
        if arr.shape[2] == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected single-channel uint mask, got shape={arr.shape} at {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got ndim={arr.ndim} at {path}")
    return arr


def build_residual(prev_mask: np.ndarray, cur_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if prev_mask.shape != cur_mask.shape:
        raise ValueError(f"shape mismatch: prev={prev_mask.shape}, cur={cur_mask.shape}")
    c = (cur_mask != prev_mask).astype(np.uint8)
    v = np.where(c == 1, cur_mask, 0).astype(cur_mask.dtype)
    return c, v


def reconstruct_from_residual(prev_mask: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.where(c.astype(bool), v, prev_mask).astype(prev_mask.dtype, copy=False)


def parse_years_from_name(name: str) -> List[int]:
    return [int(x) for x in YEAR_RE.findall(name)]


def pair_sort_key(path: Path) -> Tuple[int, str]:
    years = parse_years_from_name(path.stem)
    if years:
        return years[-1], path.name
    return 10**9, path.name


def compress_bytes(data: bytes, codec: str, level: int) -> bytes:
    codec = codec.lower()
    if codec == "lzma":
        return lzma.compress(data, preset=max(0, min(9, int(level))))
    if codec == "zstd":
        try:
            import zstandard as zstd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("zstd requested but python-zstandard is not installed") from e
        comp = zstd.ZstdCompressor(level=int(level))
        return comp.compress(data)
    raise ValueError(f"Unsupported codec: {codec}")


def pack_change_mask_bits(c: np.ndarray) -> bytes:
    flat = c.astype(np.uint8).reshape(-1)
    packed = np.packbits(flat, bitorder="little")
    return packed.tobytes()


def iter_tiles_2d(arr: np.ndarray, tile_size: int, stride: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    h, w = arr.shape[:2]
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            yield y, x, arr[y : y + tile_size, x : x + tile_size]


def tile_grid_shape(h: int, w: int, tile_size: int, stride: int) -> Tuple[int, int]:
    return ((h - tile_size) // stride + 1, (w - tile_size) // stride + 1)


def flatten_finite(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def percentile_or_nan(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))
