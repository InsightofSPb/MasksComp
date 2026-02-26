#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bz2
import csv
import lzma
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    print("ERROR: opencv-python is required (cv2). Install it in your env.", file=sys.stderr)
    raise

# Optional zstd
_ZSTD_MODE = "none"
try:
    import zstandard as zstd  # type: ignore
    _ZSTD_MODE = "py"
except Exception:
    _ZSTD_MODE = "none"


PAIR_RE = re.compile(r"(?P<y1>\d{4})_(?P<y2>\d{4})_.*\.png$", re.IGNORECASE)


def parse_pair_years(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = PAIR_RE.match(name)
    if not m:
        return None, None
    return int(m.group("y1")), int(m.group("y2"))


def as_uint_stream(arr: np.ndarray) -> Tuple[bytes, str]:
    """
    Convert mask/image to a byte stream in a stable way.
    If single-channel uint8/uint16 -> use raw bytes.
    If multi-channel -> raw bytes of the full array.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes(), str(arr.dtype)


def delta_left_stream(mask: np.ndarray) -> bytes:
    """
    Delta-encode along rows: d[x] = (v[x] - v[x-1]) mod 256 (or 65536).
    Works for uint8/uint16 single-channel.
    """
    if mask.ndim != 2:
        raise ValueError("delta_left_stream expects single-channel (H,W) array")

    if mask.dtype == np.uint8:
        mod = 256
        out = np.empty_like(mask)
        out[:, 0] = mask[:, 0]
        out[:, 1:] = (mask[:, 1:].astype(np.int16) - mask[:, :-1].astype(np.int16)) % mod
        return out.astype(np.uint8, copy=False).tobytes()

    if mask.dtype == np.uint16:
        mod = 65536
        out = np.empty_like(mask)
        out[:, 0] = mask[:, 0]
        out[:, 1:] = (mask[:, 1:].astype(np.int32) - mask[:, :-1].astype(np.int32)) % mod
        return out.astype(np.uint16, copy=False).tobytes()

    raise ValueError(f"delta_left_stream supports only uint8/uint16, got {mask.dtype}")


def _uvarint(n: int) -> bytes:
    """Unsigned varint encoding."""
    if n < 0:
        raise ValueError("varint expects non-negative")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def rle_row_stream(mask: np.ndarray) -> bytes:
    """
    Row-wise RLE for single-channel uint8/uint16 masks:
    stream = [value_bytes][len_varint] ... repeated.
    """
    if mask.ndim != 2:
        raise ValueError("rle_row_stream expects single-channel (H,W) array")
    if mask.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"rle_row_stream supports only uint8/uint16, got {mask.dtype}")

    h, w = mask.shape
    out = bytearray()
    itemsize = mask.dtype.itemsize

    for r in range(h):
        row = mask[r]
        # Ensure Python int for comparisons
        cur = int(row[0])
        run = 1
        for c in range(1, w):
            v = int(row[c])
            if v == cur:
                run += 1
            else:
                out.extend(cur.to_bytes(itemsize, "little", signed=False))
                out.extend(_uvarint(run))
                cur = v
                run = 1
        out.extend(cur.to_bytes(itemsize, "little", signed=False))
        out.extend(_uvarint(run))

    return bytes(out)


def compress_bytes(data: bytes, codec: str, level: int = 3) -> bytes:
    if codec == "zlib":
        return zlib.compress(data, level=level)
    if codec == "bz2":
        return bz2.compress(data, compresslevel=max(1, min(9, level)))
    if codec == "lzma":
        # preset 0..9 (9 is strongest)
        preset = max(0, min(9, level))
        return lzma.compress(data, preset=preset)
    if codec == "zstd":
        if _ZSTD_MODE == "py":
            c = zstd.ZstdCompressor(level=level)
            return c.compress(data)
        raise RuntimeError("zstd requested but python-zstandard is not installed")
    raise ValueError(f"Unknown codec: {codec}")


def iter_pngs(root: Path, subdirs: List[str]) -> Iterable[Tuple[str, Path]]:
    for sd in subdirs:
        d = root / sd
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")):
            yield sd, p


def read_png(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return arr


def tiles_view(arr: np.ndarray, tile: int) -> Iterable[Tuple[int, int, np.ndarray]]:
    h, w = arr.shape[:2]
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            patch = arr[y:min(h, y+tile), x:min(w, x+tile)]
            yield y, x, patch


def main():
    ap = argparse.ArgumentParser("Benchmark compression bbp for facade masks/diff maps")
    ap.add_argument("--root", required=True, help="Path to mask_adaptation/facades")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--include", default="warped_masks,diff_maps", help="Comma-separated subdirs to include")
    ap.add_argument("--codecs", default="zlib,bz2,lzma,zstd", help="Comma-separated codecs: zlib,bz2,lzma,zstd")
    ap.add_argument("--levels", default="1,3,6,9", help="Comma-separated levels (meaning depends on codec)")
    ap.add_argument("--streams", default="raw,delta_left,rle_row", help="Comma-separated: raw,delta_left,rle_row")
    ap.add_argument("--tile", type=int, default=0, help="If >0, compute per-tile bbp and report mean/p50/p90")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)

    subdirs = [s.strip() for s in args.include.split(",") if s.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    streams = [s.strip() for s in args.streams.split(",") if s.strip()]

    # Validate zstd availability
    if "zstd" in codecs and _ZSTD_MODE == "none":
        print("WARN: zstd codec requested but python-zstandard is not installed; zstd will be skipped.", file=sys.stderr)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "facade_id", "subdir", "filename", "y1", "y2",
        "H", "W", "C", "dtype", "unique_vals",
        "stream", "codec", "level",
        "bytes_in", "bytes_out", "bbp",
        "tile", "tiles_count", "bbp_mean", "bbp_p50", "bbp_p90"
    ]

    facades = [p for p in sorted(root.iterdir()) if p.is_dir()]
    rows_written = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for facade_dir in facades:
            facade_id = facade_dir.name

            for sd, png_path in iter_pngs(facade_dir, subdirs):
                y1, y2 = parse_pair_years(png_path.name)

                arr = read_png(png_path)
                # Ensure consistent shape info
                if arr.ndim == 2:
                    H, W = arr.shape
                    C = 1
                else:
                    H, W, C = arr.shape

                dtype_str = str(arr.dtype)

                unique_vals = ""
                if arr.ndim == 2 and arr.dtype in (np.uint8, np.uint16):
                    # Safe unique count (downsample if huge? but usually fine)
                    u = np.unique(arr)
                    unique_vals = str(len(u))

                # Tile mode
                if args.tile and args.tile > 0:
                    tile = args.tile
                    # For each (stream, codec, level) compute distribution across tiles
                    for stream in streams:
                        # Only apply delta/rle to single-channel integer masks
                        def make_stream_bytes(patch: np.ndarray) -> bytes:
                            if stream == "raw":
                                b, _ = as_uint_stream(patch)
                                return b
                            if stream == "delta_left":
                                return delta_left_stream(patch)
                            if stream == "rle_row":
                                return rle_row_stream(patch)
                            raise ValueError(stream)

                        # Precompute tile byte streams
                        tile_bytes: List[bytes] = []
                        tile_pixels: List[int] = []
                        for _, _, patch in tiles_view(arr, tile):
                            if stream != "raw" and (patch.ndim != 2 or patch.dtype not in (np.uint8, np.uint16)):
                                continue
                            try:
                                b = make_stream_bytes(patch)
                            except Exception:
                                continue
                            tile_bytes.append(b)
                            tile_pixels.append(int(patch.shape[0] * patch.shape[1] * (1 if patch.ndim == 2 else patch.shape[2])))

                        if not tile_bytes:
                            continue

                        for codec in codecs:
                            if codec == "zstd" and _ZSTD_MODE == "none":
                                continue
                            for level in levels:
                                bbps = []
                                bytes_in_total = 0
                                bytes_out_total = 0
                                for b, pix in zip(tile_bytes, tile_pixels):
                                    bytes_in_total += len(b)
                                    cb = compress_bytes(b, codec=codec, level=level)
                                    bytes_out_total += len(cb)
                                    bbps.append((len(cb) * 8.0) / max(1, pix))

                                bbps_np = np.array(bbps, dtype=np.float64)
                                row = dict(
                                    facade_id=facade_id,
                                    subdir=sd,
                                    filename=png_path.name,
                                    y1=y1 if y1 is not None else "",
                                    y2=y2 if y2 is not None else "",
                                    H=H, W=W, C=C,
                                    dtype=dtype_str,
                                    unique_vals=unique_vals,
                                    stream=stream,
                                    codec=codec,
                                    level=level,
                                    bytes_in=bytes_in_total,
                                    bytes_out=bytes_out_total,
                                    bbp=(bytes_out_total * 8.0) / max(1, H * W * C),
                                    tile=tile,
                                    tiles_count=len(bbps),
                                    bbp_mean=float(bbps_np.mean()),
                                    bbp_p50=float(np.quantile(bbps_np, 0.50)),
                                    bbp_p90=float(np.quantile(bbps_np, 0.90)),
                                )
                                w.writerow(row)
                                rows_written += 1

                else:
                    # Full-image mode
                    for stream in streams:
                        if stream == "raw":
                            data, _ = as_uint_stream(arr)
                        elif stream == "delta_left":
                            if arr.ndim != 2 or arr.dtype not in (np.uint8, np.uint16):
                                continue
                            data = delta_left_stream(arr)
                        elif stream == "rle_row":
                            if arr.ndim != 2 or arr.dtype not in (np.uint8, np.uint16):
                                continue
                            data = rle_row_stream(arr)
                        else:
                            raise ValueError(stream)

                        pixels = int(H * W * C)
                        for codec in codecs:
                            if codec == "zstd" and _ZSTD_MODE == "none":
                                continue
                            for level in levels:
                                comp = compress_bytes(data, codec=codec, level=level)
                                row = dict(
                                    facade_id=facade_id,
                                    subdir=sd,
                                    filename=png_path.name,
                                    y1=y1 if y1 is not None else "",
                                    y2=y2 if y2 is not None else "",
                                    H=H, W=W, C=C,
                                    dtype=dtype_str,
                                    unique_vals=unique_vals,
                                    stream=stream,
                                    codec=codec,
                                    level=level,
                                    bytes_in=len(data),
                                    bytes_out=len(comp),
                                    bbp=(len(comp) * 8.0) / max(1, pixels),
                                    tile="",
                                    tiles_count="",
                                    bbp_mean="",
                                    bbp_p50="",
                                    bbp_p90="",
                                )
                                w.writerow(row)
                                rows_written += 1

    print(f"OK: wrote {rows_written} rows to {out_csv}")
    if "zstd" in codecs and _ZSTD_MODE == "none":
        print("NOTE: zstd skipped (python-zstandard not installed).")


if __name__ == "__main__":
    main()