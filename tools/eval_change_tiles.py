#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from maskscomp.change_detection import (
    compress_bytes,
    configure_logging,
    iter_tiles_2d,
    pack_change_mask_bits,
    percentile_or_nan,
    read_mask,
    read_pairs_csv,
    set_deterministic_seed,
    tile_grid_shape,
)

LOGGER = logging.getLogger("eval_change_tiles")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute tile-wise surprise maps from residual representations")
    p.add_argument("--mode", choices=["classic_residual", "lm_residual"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--codec", type=str, default="lzma")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _estimate_probs(records, data_root: Path):
    hist_v = np.ones(256, dtype=np.float64)
    c1 = 1.0
    c0 = 1.0
    for r in records:
        c = read_mask(data_root / r.pair_id / "residual_C" / f"{Path(r.cur_path).stem}.png")
        v = read_mask(data_root / r.pair_id / "residual_V" / f"{Path(r.cur_path).stem}.png")
        c1 += float(c.sum())
        c0 += float(c.size - c.sum())
        vals = v[c > 0]
        if vals.size:
            bins = np.bincount(vals.astype(np.uint8), minlength=256)
            hist_v += bins
    p1 = c1 / (c1 + c0)
    p0 = c0 / (c1 + c0)
    pv = hist_v / hist_v.sum()
    return p0, p1, pv


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_deterministic_seed(args.seed)

    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]
    heat_root = args.out_dir / "heatmaps_tiles"
    heat_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "change_tiles_scores.csv"

    p0 = p1 = 0.5
    pv = np.ones(256, dtype=np.float64) / 256.0
    if args.mode == "lm_residual":
        p0, p1, pv = _estimate_probs(rows, args.data_root)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["split", "facade_id", "filename", "bbp", "heat_min", "heat_p50", "heat_p90", "heat_max", "heatmap_path"],
        )
        wr.writeheader()
        for r in rows:
            c = read_mask(args.data_root / r.pair_id / "residual_C" / f"{Path(r.cur_path).stem}.png")
            v = read_mask(args.data_root / r.pair_id / "residual_V" / f"{Path(r.cur_path).stem}.png")
            h, w = c.shape
            gh, gw = tile_grid_shape(h, w, args.tile_size, args.stride)
            heat = np.zeros((gh, gw), dtype=np.float32)

            idx = 0
            for y, x, c_tile in iter_tiles_2d(c, args.tile_size, args.stride):
                gy, gx = divmod(idx, gw)
                v_tile = v[y : y + args.tile_size, x : x + args.tile_size]
                if args.mode == "classic_residual":
                    bits = 8.0 * len(compress_bytes(pack_change_mask_bits(c_tile), args.codec, args.level))
                    bits += 8.0 * len(compress_bytes(v_tile.astype(np.uint8, copy=False).tobytes(), args.codec, args.level))
                else:
                    c_flat = c_tile.reshape(-1)
                    bits_c = float(-(np.log2(p1) * c_flat.sum() + np.log2(p0) * (c_flat.size - c_flat.sum())))
                    changed = v_tile[c_tile > 0].reshape(-1)
                    bits_v = float(-np.log2(pv[np.clip(changed, 0, 255)]).sum()) if changed.size else 0.0
                    bits = bits_c + bits_v
                heat[gy, gx] = bits
                idx += 1

            npy_dir = heat_root / r.sample_id
            npy_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(r.cur_path).stem
            npy_path = npy_dir / f"{stem}.npy"
            np.save(npy_path, heat)

            wr.writerow(
                {
                    "split": r.split,
                    "facade_id": r.sample_id,
                    "filename": Path(r.cur_path).name,
                    "bbp": float(heat.sum()) / max(1, h * w),
                    "heat_min": float(heat.min()),
                    "heat_p50": percentile_or_nan(heat.ravel(), 50),
                    "heat_p90": percentile_or_nan(heat.ravel(), 90),
                    "heat_max": float(heat.max()),
                    "heatmap_path": str(npy_path),
                }
            )

    LOGGER.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
