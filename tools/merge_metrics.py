#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-method metrics CSVs")
    p.add_argument("--metrics-dir", type=Path, required=True)
    p.add_argument("--glob", type=str, default="*.csv")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parts = []
    for p in sorted(args.metrics_dir.glob(args.glob)):
        if p.resolve() == args.out.resolve():
            continue
        parts.append(pd.read_csv(p))

    if not parts:
        raise SystemExit(f"No CSV files matched {args.metrics_dir / args.glob}")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
