#!/usr/bin/env python3
import argparse
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate bbp statistics by (facade_id, codec, level)."
    )
    p.add_argument(
        "--in-csv",
        required=True,
        help="Path to input classic_codecs.csv (must have header).",
    )
    p.add_argument(
        "--out",
        default="-",
        help="Output path for aggregated table. Use '-' for stdout. (default: '-')",
    )
    p.add_argument(
        "--sep",
        default=",",
        help="Output separator: ',' for CSV or '\\t' for TSV. (default: ',')",
    )
    p.add_argument(
        "--codecs",
        default="lzma,zstd",
        help="Comma-separated list of codecs to keep (case-insensitive). (default: 'lzma,zstd')",
    )
    p.add_argument(
        "--metric",
        choices=["bbp", "bbp_mean"],
        default="bbp",
        help="Which column to average. Use 'bbp' (recommended) or 'bbp_mean' if it is populated. (default: 'bbp')",
    )
    p.add_argument(
        "--dropna",
        action="store_true",
        help="Drop rows with NaN in level/metric (default: True behavior regardless).",
    )
    p.add_argument(
        "--pivot",
        action="store_true",
        help="Also print a pivoted view to stdout (levels as columns). Output file still gets the long table.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    # Basic sanity checks
    required = {"facade_id", "codec", "level", args.metric}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. موجود: {list(df.columns)}")

    # Normalize codec names
    df["codec"] = df["codec"].astype(str).str.strip().str.lower()

    codecs = [c.strip().lower() for c in args.codecs.split(",") if c.strip()]
    df = df[df["codec"].isin(codecs)].copy()

    # Numeric conversion
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")

    # Drop rows without level or metric (almost always desired)
    df = df.dropna(subset=["level", args.metric])

    if df.empty:
        # Helpful diagnostics
        avail = (
            pd.read_csv(args.in_csv)["codec"]
            .astype(str).str.strip().str.lower()
            .value_counts()
        )
        msg = (
            "No rows left after filtering/cleaning.\n"
            f"Requested codecs: {codecs}\n"
            "Top codecs in file:\n"
            f"{avail.head(30).to_string()}\n"
        )
        raise SystemExit(msg)

    out = (
        df.groupby(["facade_id", "codec", "level"], dropna=False)[args.metric]
          .mean()
          .reset_index()
          .rename(columns={args.metric: "bbp_mean" if args.metric == "bbp" else "bbp_mean_avg"})
          .sort_values(["facade_id", "codec", "level"])
    )

    # Write output
    if args.out == "-":
        out.to_csv(sys.stdout, index=False, sep=args.sep)
    else:
        out.to_csv(args.out, index=False, sep=args.sep)

    # Optional pivot print (for quick inspection)
    if args.pivot:
        pivot_val = out.columns[-1]
        pt = (
            out.pivot_table(index=["facade_id", "codec"], columns="level", values=pivot_val, aggfunc="mean")
               .sort_index()
        )
        print("\n# Pivot view (levels as columns):", file=sys.stderr)
        print(pt.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()