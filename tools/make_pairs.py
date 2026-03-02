#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from maskscomp.change_detection import (
    PairRecord,
    configure_logging,
    pair_sort_key,
    parse_years_from_name,
    read_split_ids,
    set_deterministic_seed,
    write_pairs_csv,
)

LOGGER = logging.getLogger("make_pairs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prev/cur pair lists for change detection datasets")
    p.add_argument("--dataset", choices=["a2d2", "facades"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--delta", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_pairs_for_sample(sample_id: str, images: List[Path], split: str, delta: int) -> List[PairRecord]:
    rows: List[PairRecord] = []
    ordered = sorted(images, key=pair_sort_key)
    for i in range(len(ordered) - delta):
        p_prev = ordered[i]
        p_cur = ordered[i + delta]
        years_prev = parse_years_from_name(p_prev.stem)
        years_cur = parse_years_from_name(p_cur.stem)
        t_prev = str(years_prev[-1]) if years_prev else str(i)
        t_cur = str(years_cur[-1]) if years_cur else str(i + delta)
        pair_id = f"{sample_id}__{p_prev.stem}__{p_cur.stem}"
        rows.append(
            PairRecord(
                pair_id=pair_id,
                sample_id=sample_id,
                prev_path=str(p_prev),
                cur_path=str(p_cur),
                t_prev=t_prev,
                t_cur=t_cur,
                split=split,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_deterministic_seed(args.seed)

    train_ids = set(read_split_ids(args.splits_dir, "train"))
    val_ids = set(read_split_ids(args.splits_dir, "val"))
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise ValueError(f"split leakage in ids: {sorted(list(overlap))[:5]}")

    all_rows: Dict[str, List[PairRecord]] = {"train": [], "val": []}

    for split, ids in (("train", train_ids), ("val", val_ids)):
        for sample_id in sorted(ids):
            d = args.data_root / sample_id / args.subdir
            if not d.exists():
                LOGGER.warning("Missing subdir for sample_id=%s: %s", sample_id, d)
                continue
            images = [p.relative_to(args.data_root) for p in sorted(d.glob("*.png"))]
            if len(images) < 2:
                continue
            all_rows[split].extend(make_pairs_for_sample(sample_id, images, split=split, delta=args.delta))

    for split, rows in all_rows.items():
        out_csv = args.out_dir / f"{args.dataset}_pairs_{split}.csv"
        write_pairs_csv(out_csv, rows)
        LOGGER.info("Wrote %s (%d rows)", out_csv, len(rows))


if __name__ == "__main__":
    main()
