#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bz2
import csv
import lzma
import re
import sys
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    print("ERROR: need opencv-python (cv2)", file=sys.stderr)
    raise

# optional zstd
_ZSTD_OK = False
try:
    import zstandard as zstd  # type: ignore
    _ZSTD_OK = True
except Exception:
    _ZSTD_OK = False

NAME_RE = re.compile(r"^(?P<city>.+)_(?P<seq>\d{6})_(?P<frame>\d{6})_gtFine_(?P<kind>.+)\.png$")


def parse_name(p: Path) -> Tuple[str, Optional[int], Optional[int], str]:
    m = NAME_RE.match(p.name)
    if not m:
        return "", None, None, ""
    return m.group("city"), int(m.group("seq")), int(m.group("frame")), m.group("kind")


def delta_left_stream(mask: np.ndarray) -> bytes:
    if mask.ndim != 2:
        raise ValueError("delta_left expects (H,W)")
    if mask.dtype == np.uint8:
        out = np.empty_like(mask)
        out[:, 0] = mask[:, 0]
        out[:, 1:] = (mask[:, 1:].astype(np.int16) - mask[:, :-1].astype(np.int16)) % 256
        return out.tobytes()
    if mask.dtype == np.uint16:
        out = np.empty_like(mask)
        out[:, 0] = mask[:, 0]
        out[:, 1:] = (mask[:, 1:].astype(np.int32) - mask[:, :-1].astype(np.int32)) % 65536
        return out.tobytes()
    raise ValueError(f"delta_left supports uint8/uint16, got {mask.dtype}")


def _uvarint(n: int) -> bytes:
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
    if mask.ndim != 2:
        raise ValueError("rle_row expects (H,W)")
    if mask.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"rle_row supports uint8/uint16, got {mask.dtype}")

    h, w = mask.shape
    itemsize = mask.dtype.itemsize
    out = bytearray()

    for r in range(h):
        row = mask[r]
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


def compress_bytes(data: bytes, codec: str, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(data, level=max(1, min(9, level)))
    if codec == "bz2":
        return bz2.compress(data, compresslevel=max(1, min(9, level)))
    if codec == "lzma":
        return lzma.compress(data, preset=max(0, min(9, level)))
    if codec == "zstd":
        if not _ZSTD_OK:
            raise RuntimeError("python-zstandard not installed")
        c = zstd.ZstdCompressor(level=level)
        return c.compress(data)
    raise ValueError(codec)


def tiles_view(arr: np.ndarray, tile: int):
    h, w = arr.shape[:2]
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            yield y, x, arr[y:min(h, y + tile), x:min(w, x + tile)]


def main():
    ap = argparse.ArgumentParser("Cityscapes gtFine mask compression benchmark")
    ap.add_argument("--root", required=True, help="Root containing gtFine/{train,val,test}")
    ap.add_argument("--split", default="train,val", help="Comma-separated: train,val,test")
    ap.add_argument("--pattern", default="*_gtFine_labelIds.png", help="Glob pattern for masks")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--codecs", default="zlib,bz2,lzma", help="Comma-separated: zlib,bz2,lzma,zstd")
    ap.add_argument("--levels", default="1,3,6,9", help="Comma-separated levels")
    ap.add_argument("--streams", default="raw,delta_left,rle_row", help="Comma-separated: raw,delta_left,rle_row")
    ap.add_argument("--tile", type=int, default=0, help="If >0 compute tile bbp stats (mean/p50/p90)")
    ap.add_argument("--max-files", type=int, default=0, help="If >0 limit number of files (for quick test)")
    args = ap.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    streams = [s.strip() for s in args.streams.split(",") if s.strip()]

    if "zstd" in codecs and not _ZSTD_OK:
        print("WARN: zstd requested but python-zstandard not installed; skipping zstd.", file=sys.stderr)
        codecs = [c for c in codecs if c != "zstd"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split", "city", "seq", "frame", "kind", "path",
        "H", "W", "dtype", "unique_vals",
        "stream", "codec", "level",
        "bytes_in", "bytes_out", "bbp",
        "tile", "tiles_count", "bbp_mean", "bbp_p50", "bbp_p90",
    ]

    files: List[Tuple[str, Path]] = []
    for sp in splits:
        sp_dir = root / sp
        if not sp_dir.exists():
            continue
        for p in sorted(sp_dir.rglob(args.pattern)):
            files.append((sp, p))

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for sp, p in files:
            arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            if arr.ndim != 2:
                # Cityscapes labelIds should be 1-channel; if not, skip
                continue

            H, W = arr.shape
            dtype = str(arr.dtype)
            unique_vals = int(len(np.unique(arr))) if arr.dtype in (np.uint8, np.uint16) else ""

            city, seq, frame, kind = parse_name(p)

            if args.tile and args.tile > 0:
                tile = args.tile
                for stream in streams:
                    tile_bytes = []
                    tile_pix = []
                    for _, _, patch in tiles_view(arr, tile):
                        try:
                            if stream == "raw":
                                b = np.ascontiguousarray(patch).tobytes()
                            elif stream == "delta_left":
                                b = delta_left_stream(patch)
                            elif stream == "rle_row":
                                b = rle_row_stream(patch)
                            else:
                                raise ValueError(stream)
                        except Exception:
                            continue
                        tile_bytes.append(b)
                        tile_pix.append(int(patch.shape[0] * patch.shape[1]))

                    if not tile_bytes:
                        continue

                    for codec in codecs:
                        for level in levels:
                            bbps = []
                            bytes_in_total = 0
                            bytes_out_total = 0
                            for b, pix in zip(tile_bytes, tile_pix):
                                bytes_in_total += len(b)
                                cb = compress_bytes(b, codec, level)
                                bytes_out_total += len(cb)
                                bbps.append((len(cb) * 8.0) / max(1, pix))

                            bbps_np = np.array(bbps, dtype=np.float64)
                            w.writerow(dict(
                                split=sp, city=city, seq=seq or "", frame=frame or "", kind=kind, path=str(p),
                                H=H, W=W, dtype=dtype, unique_vals=unique_vals,
                                stream=stream, codec=codec, level=level,
                                bytes_in=bytes_in_total, bytes_out=bytes_out_total,
                                bbp=(bytes_out_total * 8.0) / max(1, H * W),
                                tile=tile, tiles_count=len(bbps),
                                bbp_mean=float(bbps_np.mean()),
                                bbp_p50=float(np.quantile(bbps_np, 0.50)),
                                bbp_p90=float(np.quantile(bbps_np, 0.90)),
                            ))
                            rows += 1
            else:
                for stream in streams:
                    try:
                        if stream == "raw":
                            data = np.ascontiguousarray(arr).tobytes()
                        elif stream == "delta_left":
                            data = delta_left_stream(arr)
                        elif stream == "rle_row":
                            data = rle_row_stream(arr)
                        else:
                            raise ValueError(stream)
                    except Exception:
                        continue

                    for codec in codecs:
                        for level in levels:
                            comp = compress_bytes(data, codec, level)
                            w.writerow(dict(
                                split=sp, city=city, seq=seq or "", frame=frame or "", kind=kind, path=str(p),
                                H=H, W=W, dtype=dtype, unique_vals=unique_vals,
                                stream=stream, codec=codec, level=level,
                                bytes_in=len(data), bytes_out=len(comp),
                                bbp=(len(comp) * 8.0) / max(1, H * W),
                                tile="", tiles_count="", bbp_mean="", bbp_p50="", bbp_p90="",
                            ))
                            rows += 1

    print(f"OK: wrote {rows} rows to {out_path} (files={len(files)})")


if __name__ == "__main__":
    main()