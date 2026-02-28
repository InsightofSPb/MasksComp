#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from maskscomp.entropy_coding.lm_codec import OnlineConfig, encode_file
from maskscomp.lm_entropy import collect_images, load_split_list, read_mask_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pretrained/online x ideal/actual codec matrix"
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=Path("output/lm_matrix_summary.csv"))
    p.add_argument("--tmp-dir", type=Path, default=Path("output/lm_matrix_bins"))
    p.add_argument("--use-2d-context", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--online-lr", type=float, default=1e-3)
    p.add_argument("--online-steps-per-row", type=int, default=1)
    p.add_argument("--online-clip", type=float, default=1.0)
    p.add_argument("--online-after", choices=["row", "image"], default="row")
    return p.parse_args()


def summarize(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 90))


def _blank_actual_len_fields(row: dict) -> dict:
    row["bbp_len_median"] = ""
    row["bbp_len_p90"] = ""
    row["len_frac_median"] = ""
    row["len_frac_p90"] = ""
    return row


def main() -> None:
    args = parse_args()
    recs = collect_images(args.data_root, subdir=args.subdir)
    splits = {
        "train": set(load_split_list(args.splits_dir / "facade_train.txt")),
        "val": set(load_split_list(args.splits_dir / "facade_val.txt")),
    }
    modes = {
        "pretrained": OnlineConfig(mode="pretrained"),
        "online": OnlineConfig(
            mode="online",
            lr=args.online_lr,
            steps_per_row=args.online_steps_per_row,
            clip=args.online_clip,
            online_after=args.online_after,
        ),
    }

    rows_out = []
    for split, facades in splits.items():
        split_recs = [r for r in recs if r.facade_id in facades]
        for mode_name, online in modes.items():
            ideal_bbp_total, ideal_bbp_len, ideal_len_frac = [], [], []
            actual_bbp_total = []
            overheads = []
            header_overheads = []
            payload_bits = []
            with_header_bits = []

            for rec in split_recs:
                mask = read_mask_png(rec.path)
                if mask is None:
                    continue
                pixels = max(1, int(mask.shape[0] * mask.shape[1]))
                out_bin = (
                    args.tmp_dir
                    / mode_name
                    / split
                    / Path(rec.rel_path).with_suffix(".bin")
                )
                stats = encode_file(
                    rec.path,
                    args.checkpoint,
                    out_bin,
                    args.use_2d_context,
                    online,
                    torch.device(args.device),
                    None,
                )

                ideal_bbp_total.append(stats["ideal_bits"] / pixels)
                ideal_bbp_len.append(stats["ideal_len_bits"] / pixels)
                if stats["ideal_bits"] > 0:
                    ideal_len_frac.append(stats["ideal_len_bits"] / stats["ideal_bits"])

                actual_bbp_total.append(stats["actual_bits_payload"] / pixels)
                overheads.append(stats["overhead_pct"])
                header_overheads.append(stats["actual_header_overhead_pct"])
                payload_bits.append(stats["actual_bits_payload"])
                with_header_bits.append(stats["actual_bits_with_header"])

            ideal_total_med, ideal_total_p90 = summarize(ideal_bbp_total)
            ideal_len_med, ideal_len_p90 = summarize(ideal_bbp_len)
            len_frac_med, len_frac_p90 = summarize(ideal_len_frac)
            rows_out.append(
                {
                    "mode": mode_name,
                    "measurement": "ideal",
                    "split": split,
                    "bbp_total_median": ideal_total_med,
                    "bbp_total_p90": ideal_total_p90,
                    "bbp_len_median": ideal_len_med,
                    "bbp_len_p90": ideal_len_p90,
                    "len_frac_median": len_frac_med,
                    "len_frac_p90": len_frac_p90,
                    "overhead_pct": "",
                    "actual_header_overhead_pct": "",
                    "actual_bits_payload_median": "",
                    "actual_bits_with_header_median": "",
                }
            )

            actual_total_med, actual_total_p90 = summarize(actual_bbp_total)
            payload_med, _ = summarize(payload_bits)
            with_header_med, _ = summarize(with_header_bits)
            row_actual = {
                "mode": mode_name,
                "measurement": "actual",
                "split": split,
                "bbp_total_median": actual_total_med,
                "bbp_total_p90": actual_total_p90,
                "overhead_pct": float(
                    np.median(np.asarray(overheads, dtype=np.float64))
                ),
                "actual_header_overhead_pct": float(
                    np.median(np.asarray(header_overheads, dtype=np.float64))
                ),
                "actual_bits_payload_median": payload_med,
                "actual_bits_with_header_median": with_header_med,
            }
            rows_out.append(_blank_actual_len_fields(row_actual))

    fieldnames = [
        "mode",
        "measurement",
        "split",
        "bbp_total_median",
        "bbp_total_p90",
        "bbp_len_median",
        "bbp_len_p90",
        "len_frac_median",
        "len_frac_p90",
        "overhead_pct",
        "actual_header_overhead_pct",
        "actual_bits_payload_median",
        "actual_bits_with_header_median",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows_out)


if __name__ == "__main__":
    main()
