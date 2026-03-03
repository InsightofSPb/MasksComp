#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from maskscomp.change_detection import read_pairs_csv, tile_grid_shape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MSDZip run-level bits dumps to tile heatmaps")
    p.add_argument("--run-bits-dir", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--tile-size", type=int, required=True)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ids-txt", type=Path, default=None)
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def _tile_heat_from_density(density: np.ndarray, tile_size: int, stride: int) -> np.ndarray:
    h, w = density.shape
    gh, gw = tile_grid_shape(h, w, tile_size, stride)
    integ = np.pad(density.astype(np.float64, copy=False), ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    heat = np.zeros((gh, gw), dtype=np.float32)
    area = float(tile_size * tile_size)
    for gy in range(gh):
        y0 = gy * stride
        y1 = y0 + tile_size
        x0 = np.arange(gw, dtype=np.int64) * stride
        x1 = x0 + tile_size
        sums = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        heat[gy, :] = (sums / area).astype(np.float32)
    return heat


def _load_ids(ids_txt: Path) -> set[str]:
    ids = set()
    for ln in ids_txt.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            ids.add(s)
    return ids


def main() -> None:
    args = parse_args()
    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]

    if args.ids_txt is not None:
        keep = _load_ids(args.ids_txt)
        rows = [r for r in rows if str(r.pair_id) in keep]
    if args.max_items is not None:
        rows = rows[: int(args.max_items)]

    heat_root = args.out_dir / "heatmaps_tiles"
    heat_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "change_tiles_scores.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "facade_id",
                "filename",
                "bbp",
                "heat_p90",
                "heat_max",
                "heat_p50",
                "heat_min",
                "heatmap_path",
            ],
        )
        wr.writeheader()

        for r in rows:
            stem = Path(r.cur_path).stem
            src = args.run_bits_dir / r.pair_id / f"{stem}.npz"
            if not src.exists():
                continue

            with np.load(src, allow_pickle=False) as z:
                h = int(np.asarray(z["H"]).reshape(-1)[0])
                w = int(np.asarray(z["W"]).reshape(-1)[0])
                y = z["y"].astype(np.int64, copy=False)
                start = z["start"].astype(np.int64, copy=False)
                length = z["length"].astype(np.int64, copy=False)
                bits_run = z["bits_run"].astype(np.float64, copy=False)

            density = np.zeros((h, w), dtype=np.float32)
            for yy, xx, ll, bb in zip(y, start, length, bits_run):
                if ll <= 0:
                    continue
                x2 = min(w, int(xx + ll))
                if 0 <= yy < h and 0 <= xx < w and x2 > xx:
                    density[int(yy), int(xx):x2] += float(bb) / float(ll)

            heat = _tile_heat_from_density(density, args.tile_size, args.stride)
            bits_total = float(bits_run.sum())
            bbp = bits_total / max(1, h * w)

            out_dir = heat_root / r.pair_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_npy = out_dir / f"{stem}.npy"
            np.save(out_npy, heat)

            wr.writerow(
                {
                    "split": args.split,
                    "facade_id": r.pair_id,
                    "filename": f"{stem}.png",
                    "bbp": bbp,
                    "heat_p90": float(np.percentile(heat, 90)),
                    "heat_max": float(np.max(heat)),
                    "heat_p50": float(np.percentile(heat, 50)),
                    "heat_min": float(np.min(heat)),
                    "heatmap_path": str(out_npy.relative_to(args.out_dir)),
                }
            )

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
