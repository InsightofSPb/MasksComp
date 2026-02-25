#!/usr/bin/env python3
"""Train/evaluate a static byte-level Markov model of order 2.

The model is trained on a byte stream and evaluated on a test byte stream with
Laplace/Dirichlet smoothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a static 2nd-order byte Markov model on train-stream and "
            "compute cross-entropy on test-stream."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-bin", type=Path, required=True, help="Training .bin stream path")
    parser.add_argument(
        "--train-stats", type=Path, required=True, help="Training stats JSON path (checked only)"
    )
    parser.add_argument("--test-bin", type=Path, required=True, help="Test .bin stream path")
    parser.add_argument("--test-stats", type=Path, required=True, help="Test stats JSON path")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet/Laplace smoothing alpha (must be > 0)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/results"),
        help="Directory where markov2_xent.csv is written",
    )
    return parser.parse_args()


def ensure_readable_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def load_stats(path: Path, label: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [k for k in ("total_pixels", "total_tokens") if k not in data]
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")

    total_pixels = int(data["total_pixels"])
    total_tokens = int(data["total_tokens"])
    if total_pixels <= 0:
        raise ValueError(f"{label} total_pixels must be > 0, got {total_pixels}")
    if total_tokens <= 0:
        raise ValueError(f"{label} total_tokens must be > 0, got {total_tokens}")

    return {"total_pixels": total_pixels, "total_tokens": total_tokens}


def train_markov2(train_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((256, 256, 256), dtype=np.uint32)
    totals = np.zeros((256, 256), dtype=np.uint32)

    prev2 = 0
    prev1 = 0

    for b in train_bytes:
        counts[prev2, prev1, b] += 1
        totals[prev2, prev1] += 1
        prev2, prev1 = prev1, int(b)

    return counts, totals


def evaluate_xent(test_bytes: np.ndarray, counts: np.ndarray, totals: np.ndarray, alpha: float) -> float:
    prev2 = 0
    prev1 = 0
    sum_bits = 0.0
    denom_add = 256.0 * alpha

    for b in test_bytes:
        c = float(counts[prev2, prev1, b])
        t = float(totals[prev2, prev1])
        p = (c + alpha) / (t + denom_add)
        sum_bits += -math.log2(p)
        prev2, prev1 = prev1, int(b)

    return sum_bits


def write_results_csv(
    out_csv: Path,
    dataset: str,
    alpha: float,
    bits_per_byte: float,
    est_compressed_bytes: float,
    bits_per_pixel: float,
    bits_per_token: float,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "alpha",
                "bits_per_byte",
                "est_compressed_bytes",
                "bits_per_pixel",
                "bits_per_token",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset,
                "alpha": alpha,
                "bits_per_byte": bits_per_byte,
                "est_compressed_bytes": est_compressed_bytes,
                "bits_per_pixel": bits_per_pixel,
                "bits_per_token": bits_per_token,
            }
        )


def main() -> None:
    args = parse_args()

    if args.alpha <= 0:
        raise ValueError(f"--alpha must be > 0, got {args.alpha}")

    train_bin = ensure_readable_file(args.train_bin, "--train-bin")
    train_stats_path = ensure_readable_file(args.train_stats, "--train-stats")
    test_bin = ensure_readable_file(args.test_bin, "--test-bin")
    test_stats_path = ensure_readable_file(args.test_stats, "--test-stats")

    _ = load_stats(train_stats_path, "train stats")
    test_stats = load_stats(test_stats_path, "test stats")

    train_bytes = np.fromfile(train_bin, dtype=np.uint8)
    test_bytes = np.fromfile(test_bin, dtype=np.uint8)

    if train_bytes.size == 0:
        raise ValueError(f"Training byte stream is empty: {train_bin}")
    if test_bytes.size == 0:
        raise ValueError(f"Test byte stream is empty: {test_bin}")

    counts, totals = train_markov2(train_bytes)
    sum_bits = evaluate_xent(test_bytes, counts, totals, args.alpha)

    bits_per_byte = float(sum_bits / test_bytes.size)
    est_compressed_bytes = float(sum_bits / 8.0)
    bits_per_pixel = float(sum_bits / test_stats["total_pixels"])
    bits_per_token = float(sum_bits / test_stats["total_tokens"])

    out_csv = args.out_dir.expanduser().resolve() / "markov2_xent.csv"
    dataset = test_bin.stem
    write_results_csv(
        out_csv=out_csv,
        dataset=dataset,
        alpha=args.alpha,
        bits_per_byte=bits_per_byte,
        est_compressed_bytes=est_compressed_bytes,
        bits_per_pixel=bits_per_pixel,
        bits_per_token=bits_per_token,
    )

    print(f"dataset={dataset}")
    print(f"alpha={args.alpha}")
    print(f"bits_per_byte={bits_per_byte:.6f}")
    print(f"est_compressed_bytes={est_compressed_bytes:.3f}")
    print(f"bits_per_pixel={bits_per_pixel:.6f}")
    print(f"bits_per_token={bits_per_token:.6f}")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
