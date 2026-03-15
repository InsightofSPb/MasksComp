#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LM proxy bits vs PNG baseline on identical raw temporal tiles")
    p.add_argument("--tile-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.tile_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No rows in tile CSV")

    lm = np.asarray([float(r["lm_bits"]) for r in rows], dtype=np.float64)
    png = np.asarray([float(r["png_bits"]) for r in rows], dtype=np.float64)
    delta = lm - png

    out = {
        "n_tiles": int(len(rows)),
        "lm_bits_mean": float(np.mean(lm)),
        "png_bits_mean": float(np.mean(png)),
        "delta_bits_mean": float(np.mean(delta)),
        "delta_bits_median": float(np.median(delta)),
        "lm_lt_png_frac": float(np.mean(lm < png)),
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out.keys()))
        wr.writeheader()
        wr.writerow(out)

    print(out)


if __name__ == "__main__":
    main()
