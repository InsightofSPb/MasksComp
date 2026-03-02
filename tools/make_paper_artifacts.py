#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper-ready tables and top-K figures")
    p.add_argument("--uncond-csv", type=Path, default=None)
    p.add_argument("--cond-csv", type=Path, default=None)
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--tiles-csv", type=Path, default=None)
    p.add_argument("--heat-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--render-topk", type=int, default=0)
    return p.parse_args()


def _write_tex_table(df: pd.DataFrame, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x), caption=caption)
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    paper_tables = args.out_dir / "_paper_tables"
    paper_figs = args.out_dir / "_paper_figs"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figs.mkdir(parents=True, exist_ok=True)

    if args.uncond_csv and args.uncond_csv.exists():
        _write_tex_table(pd.read_csv(args.uncond_csv), paper_tables / "table_uncond_bpp.tex", "Unconditional bpp")

    if args.cond_csv and args.metrics_csv and args.cond_csv.exists() and args.metrics_csv.exists():
        cond = pd.read_csv(args.cond_csv)
        met = pd.read_csv(args.metrics_csv)
        merged = cond.copy()
        for col in met.columns:
            if col not in merged.columns:
                merged[col] = met.iloc[0][col]
        _write_tex_table(merged, paper_tables / "table_cond_bpp_auc.tex", "Conditional bpp and change metrics")

    if args.render_topk > 0 and args.tiles_csv and args.heat_root:
        out = paper_figs / "topk"
        cmd = [
            "python",
            "tools/render_topk_from_csv.py",
            "--csv",
            str(args.tiles_csv),
            "--heat-root",
            str(args.heat_root),
            "--out-dir",
            str(out),
            "--topk",
            str(args.render_topk),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
