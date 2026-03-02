#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from maskscomp.change_detection import (
    compress_bytes,
    configure_logging,
    pack_change_mask_bits,
    read_mask,
    read_split_ids,
    set_deterministic_seed,
)

LOGGER = logging.getLogger("bench_residual_codecs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark conditional residual coding with LZMA/Zstd")
    p.add_argument("--data-root", type=Path, required=True, help="Residual dataset root")
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--codecs", type=str, default="lzma,zstd")
    p.add_argument("--levels", type=str, default="1,3,6,9")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_deterministic_seed(args.seed)

    pair_ids = read_split_ids(args.splits_dir, args.split)
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pair_id", "split", "H", "W", "bits_C", "bits_V", "bits_sum", "bbp_new_sum", "codec", "level"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for pair_id in pair_ids:
            base = args.data_root / pair_id
            c_paths = sorted((base / "residual_C").glob("*.png"))
            v_paths = sorted((base / "residual_V").glob("*.png"))
            if not c_paths or not v_paths:
                LOGGER.warning("Missing residual png(s) for pair_id=%s", pair_id)
                continue
            c = read_mask(c_paths[0])
            v = read_mask(v_paths[0])
            h, w = c.shape
            c_stream = pack_change_mask_bits(c)
            v_stream = v.astype("uint8", copy=False).tobytes()

            for codec in codecs:
                for level in levels:
                    bits_c = 8.0 * len(compress_bytes(c_stream, codec, level))
                    bits_v = 8.0 * len(compress_bytes(v_stream, codec, level))
                    bits_sum = bits_c + bits_v
                    wr.writerow(
                        {
                            "pair_id": pair_id,
                            "split": args.split,
                            "H": h,
                            "W": w,
                            "bits_C": bits_c,
                            "bits_V": bits_v,
                            "bits_sum": bits_sum,
                            "bbp_new_sum": bits_sum / max(1, h * w),
                            "codec": codec,
                            "level": level,
                        }
                    )
    LOGGER.info("Wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
