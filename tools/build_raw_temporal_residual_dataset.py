#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from maskscomp.raw_temporal import (
    align_prev_to_cur,
    iter_tiles,
    modular_residual,
    png_size_bytes,
    read_homography,
    read_pairs_csv,
    read_rgb,
    serialize_tile,
    write_rgb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build raw temporal residual tile dataset")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--min-valid-fraction", type=float, default=0.5)
    p.add_argument("--serialize", choices=["interleaved", "planar"], default="interleaved")
    p.add_argument("--png-level", type=int, default=6)
    p.add_argument("--write-viz", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _read_valid_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read valid mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "tiles").mkdir(parents=True, exist_ok=True)

    pairs = read_pairs_csv(args.pairs_csv)
    tile_rows = []

    split_map: dict[str, list[str]] = {}
    for rec in pairs:
        prev = read_rgb(args.data_root / rec.prev_path)
        cur = read_rgb(args.data_root / rec.cur_path)
        hh = read_homography(args.data_root / rec.homography_path) if rec.homography_path else None
        prev_aligned = align_prev_to_cur(prev, cur.shape[:2], hh)
        residual = modular_residual(cur, prev_aligned)

        valid_mask = None
        if rec.valid_mask_path:
            valid_mask = _read_valid_mask(args.data_root / rec.valid_mask_path, cur.shape[:2])

        pair_dir = args.out_root / "pairs" / rec.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        write_rgb(pair_dir / "cur_rgb.png", cur)
        write_rgb(pair_dir / "prev_aligned_rgb.png", prev_aligned)
        if args.write_viz:
            write_rgb(pair_dir / "residual_mod_rgb.png", residual)

        seqs = []
        yx = []
        valid_fracs = []
        png_bytes = []
        for y, x, tile, frac in iter_tiles(
            residual,
            tile_size=args.tile_size,
            stride=args.stride,
            valid_mask=valid_mask,
            min_valid_fraction=args.min_valid_fraction,
        ):
            seq = serialize_tile(tile, mode=args.serialize)
            seqs.append(seq)
            yx.append((y, x))
            valid_fracs.append(frac)
            cur_tile = cur[y : y + args.tile_size, x : x + args.tile_size]
            png_bytes.append(png_size_bytes(cur_tile, compress_level=args.png_level))
            tile_rows.append(
                {
                    "pair_id": rec.pair_id,
                    "sample_id": rec.sample_id,
                    "split": rec.split,
                    "tile_y": y,
                    "tile_x": x,
                    "tile_h": args.tile_size,
                    "tile_w": args.tile_size,
                    "valid_fraction": frac,
                    "png_bytes": png_bytes[-1],
                }
            )

        if seqs:
            np.savez_compressed(
                args.out_root / "tiles" / f"{rec.pair_id}.npz",
                sequences=np.stack(seqs).astype(np.uint8),
                tile_yx=np.asarray(yx, dtype=np.int32),
                valid_fraction=np.asarray(valid_fracs, dtype=np.float32),
                png_bytes=np.asarray(png_bytes, dtype=np.int32),
                tile_size=np.int32(args.tile_size),
                stride=np.int32(args.stride),
                serialize_mode=np.array(args.serialize),
                H=np.int32(cur.shape[0]),
                W=np.int32(cur.shape[1]),
            )

        split_map.setdefault(rec.split, []).append(rec.pair_id)

    with (args.out_root / "tile_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()) if tile_rows else ["pair_id"])
        wr.writeheader()
        if tile_rows:
            wr.writerows(tile_rows)

    splits_dir = args.out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, pair_ids in split_map.items():
        (splits_dir / f"facade_{split}.txt").write_text("\n".join(sorted(set(pair_ids))) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
