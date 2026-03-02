#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from maskscomp.change_detection import (
    build_residual,
    configure_logging,
    read_mask,
    read_pairs_csv,
    reconstruct_from_residual,
    set_deterministic_seed,
)

LOGGER = logging.getLogger("build_residual_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build residual_C / residual_V dataset from prev-cur pairs")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--write-png", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--verify-count", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_deterministic_seed(args.seed)

    pairs = read_pairs_csv(args.pairs_csv)
    verify_budget = int(args.verify_count)

    split_name = pairs[0].split if pairs else "train"
    splits_dir = args.out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"facade_{split_name}.txt"

    written_pair_ids = []
    for idx, row in enumerate(pairs):
        prev = read_mask(args.data_root / row.prev_path)
        cur = read_mask(args.data_root / row.cur_path)
        c, v = build_residual(prev, cur)

        base = args.out_root / row.pair_id
        c_dir = base / "residual_C"
        v_dir = base / "residual_V"
        c_dir.mkdir(parents=True, exist_ok=True)
        v_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{Path(row.cur_path).stem}.png"
        c_path = c_dir / out_name
        v_path = v_dir / out_name

        if (c_path.exists() or v_path.exists()) and not args.overwrite:
            LOGGER.info("Skipping existing pair_id=%s", row.pair_id)
        elif args.write_png:
            cv2.imwrite(str(c_path), c)
            cv2.imwrite(str(v_path), v)

        if idx < verify_budget:
            recon = reconstruct_from_residual(prev, c, v)
            if not (recon.shape == cur.shape and (recon == cur).all()):
                raise AssertionError(f"Residual reconstruction failed for {row.pair_id}")
        written_pair_ids.append(row.pair_id)

    split_path.write_text("\n".join(sorted(set(written_pair_ids))) + ("\n" if written_pair_ids else ""), encoding="utf-8")
    LOGGER.info("Wrote split file: %s", split_path)


if __name__ == "__main__":
    main()
