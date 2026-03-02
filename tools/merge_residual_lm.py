#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LM eval CSVs for residual_C and residual_V")
    p.add_argument("--csv-c", type=Path, required=True)
    p.add_argument("--csv-v", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def _prepare(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = df.copy()
    out["pair_id"] = out["facade_id"].astype(str)
    out = out[["pair_id", "bits_total", "bbp_total"]]
    out = out.rename(columns={"bits_total": f"bits_{suffix}", "bbp_total": f"bbp_{suffix}"})
    return out


def main() -> None:
    args = parse_args()
    c = _prepare(pd.read_csv(args.csv_c), "C")
    v = _prepare(pd.read_csv(args.csv_v), "V")
    m = c.merge(v, on="pair_id", how="inner", validate="one_to_one")
    m["bits_sum"] = m["bits_C"] + m["bits_V"]
    m["bbp_sum"] = m["bbp_C"] + m["bbp_V"]
    m["split"] = args.split
    m = m[["pair_id", "split", "bbp_C", "bbp_V", "bbp_sum", "bits_C", "bits_V", "bits_sum"]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} ({len(m)} rows)")


if __name__ == "__main__":
    main()
