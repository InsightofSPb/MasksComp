#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def q(x, p):
    return float(x.quantile(p)) if len(x) else float("nan")


def med(x):
    return float(x.median()) if len(x) else float("nan")


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def summarize_one(csv_path: Path, topk: int = 5):
    df = pd.read_csv(csv_path)

    # Normalize numeric columns that may exist
    for c in ["bbp", "bbp_len", "bbp_label", "len_frac_bits",
              "heat_p90", "heat_p50", "heat_max", "heat_min"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    rows = []
    for split in ["train", "val"]:
        d = df[df["split"] == split].copy()
        if d.empty:
            continue

        r = {
            "run": csv_path.parent.name,
            "csv": str(csv_path),
            "split": split,
            "n_files": int(len(d)),
            "bbp_med": med(d["bbp"]),
            "bbp_p90": q(d["bbp"], 0.90),
            "bbp_len_med": med(d["bbp_len"]),
            "bbp_label_med": med(d["bbp_label"]),
            "len_frac_med": med(d["len_frac_bits"]),
        }

        if "heat_p90" in d.columns and d["heat_p90"].notna().any():
            r.update({
                "heat_p90_med": med(d["heat_p90"]),
                "heat_p90_p90": q(d["heat_p90"], 0.90),
            })
        else:
            r.update({
                "heat_p90_med": float("nan"),
                "heat_p90_p90": float("nan"),
            })

        # Store key hyperparams if present
        for c in ["len_model", "beta", "alpha", "order",
                  "len_backoff_pair", "len_backoff_label",
                  "len_backoff_xbin", "len_xbins",
                  "len_backoff_above", "len_above_lenbins",
                  "reset_ctx_per_row"]:
            if c in d.columns:
                v = d[c].iloc[0]
                r[c] = v

        rows.append(r)

        if split == "val":
            # top-k by bbp
            cols = [c for c in ["facade_id", "filename", "bbp", "bbp_len", "bbp_label", "len_frac_bits", "heat_p90"] if c in d.columns]
            top_bbp = d.sort_values("bbp", ascending=False).head(topk)[cols]
            top_heat = None
            if "heat_p90" in d.columns and d["heat_p90"].notna().any():
                top_heat = d.sort_values("heat_p90", ascending=False).head(topk)[cols]
            return rows, top_bbp, top_heat

    return rows, None, None


def main():
    ap = argparse.ArgumentParser("Summarize ngram_mask_entropy runs (find all ngram_bbp.csv)")
    ap.add_argument("--root", default="output", help="Root folder containing run subfolders")
    ap.add_argument("--pattern", default="ngram_bbp.csv", help="Filename to search for")
    ap.add_argument("--out", default="output/_summary/summary.csv", help="Where to write summary CSV")
    ap.add_argument("--topk", type=int, default=5, help="Top-K hard examples to print per run (val)")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(root.rglob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No '{args.pattern}' found under {root}")

    all_rows = []
    for p in csv_files:
        rows, top_bbp, top_heat = summarize_one(p, topk=args.topk)
        all_rows.extend(rows)

        # Print run headline + hard cases
        run_name = p.parent.name
        print(f"\n=== {run_name} ===")
        if top_bbp is not None:
            print("[val top by bbp]")
            print(top_bbp.to_string(index=False))
        if top_heat is not None:
            print("[val top by heat_p90]")
            print(top_heat.to_string(index=False))

    summary = pd.DataFrame(all_rows)
    # nicer ordering
    key_cols = ["run", "split", "n_files", "bbp_med", "bbp_p90", "bbp_len_med", "bbp_label_med", "len_frac_med", "heat_p90_med", "heat_p90_p90"]
    other_cols = [c for c in summary.columns if c not in key_cols]
    summary = summary[key_cols + other_cols].sort_values(["run", "split"])

    summary.to_csv(out_path, index=False)
    print(f"\n[OK] wrote summary: {out_path}")

    # Print best runs by val median/p90
    val = summary[summary["split"] == "val"].copy()
    if not val.empty:
        best_med = val.sort_values("bbp_med").head(10)[["run", "bbp_med", "bbp_p90", "bbp_len_med", "bbp_label_med", "len_frac_med"]]
        best_p90 = val.sort_values("bbp_p90").head(10)[["run", "bbp_med", "bbp_p90", "bbp_len_med", "bbp_label_med", "len_frac_med"]]
        print("\n[Best by val median bbp]")
        print(best_med.to_string(index=False))
        print("\n[Best by val p90 bbp]")
        print(best_p90.to_string(index=False))


if __name__ == "__main__":
    main()