#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pairs-consecutive",
        type=Path,
        default=Path("/home/sasha/LPOSS/datasets/SPb_facades/masks_with_years/pairs_consecutive.csv"),
        help="CSV with columns: pair_id,facade_id,year_a,year_b,mask_a,mask_b,...",
    )
    p.add_argument(
        "--warp-root",
        type=Path,
        default=Path("/home/sasha/MasksComp/datasets/mask_adaptation/facades"),
        help="Root where <facade_id>/warped_masks/<year_a>_<year_b>_warped.png exists",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/sasha/MasksComp/output/facades_pairs/facades_pairs_gtwarped_gtcur.csv"),
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("/home/sasha/MasksComp/results_facades_conditional/all/splits_gt"),
    )
    p.add_argument(
        "--require-warp",
        action="store_true",
        help="If set, keep only rows where warped_masks/<ya>_<yb>_warped.png exists. Recommended.",
    )
    p.add_argument(
        "--allow-unwarped-fallback",
        action="store_true",
        help="If set and warped mask is missing, fallback to mask_a as prev_path (NOT recommended for change metrics).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.pairs_consecutive)
    need = ["pair_id", "facade_id", "year_a", "year_b", "mask_a", "mask_b"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"pairs_consecutive missing columns: {missing}. columns={list(df.columns)}")

    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df["facade_id"] = df["facade_id"].astype(str)
    df["year_a"] = pd.to_numeric(df["year_a"], errors="coerce")
    df["year_b"] = pd.to_numeric(df["year_b"], errors="coerce")
    df = df.dropna(subset=["year_a", "year_b"]).copy()
    df["year_a"] = df["year_a"].astype(int)
    df["year_b"] = df["year_b"].astype(int)

    out_rows = []
    miss_warp = 0
    miss_cur = 0
    miss_prev = 0

    for _, r in df.iterrows():
        pid = r["pair_id"]
        fid = r["facade_id"]
        ya = int(r["year_a"])
        yb = int(r["year_b"])

        # cur GT mask (year_b)
        cur = Path(str(r["mask_b"]))
        if not cur.exists():
            miss_cur += 1
            continue

        # prev: prefer warped year_a->year_b mask from MasksComp mask_adaptation
        prev_warp = args.warp_root / fid / "warped_masks" / f"{ya}_{yb}_warped.png"

        if prev_warp.exists():
            prev = prev_warp
        else:
            if args.require_warp and not args.allow_unwarped_fallback:
                miss_warp += 1
                continue
            if args.allow_unwarped_fallback:
                prev = Path(str(r["mask_a"]))
                if not prev.exists():
                    miss_prev += 1
                    continue
            else:
                miss_warp += 1
                continue

        out_rows.append(
            {
                "pair_id": pid,
                "facade_id": fid,
                "year_a": ya,
                "year_b": yb,
                "prev_path": str(prev),
                "cur_path": str(cur),
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise SystemExit(
            "No pairs produced. Check: warped_masks exist? mask_b paths exist? "
            "Try without --require-warp or enable --allow-unwarped-fallback for debugging."
        )

    # split: last year_b per facade -> val
    out["split"] = "train"
    idx = out.groupby("facade_id")["year_b"].idxmax()
    out.loc[idx, "split"] = "val"

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    train_ids = sorted(out[out["split"] == "train"]["pair_id"].tolist())
    val_ids = sorted(out[out["split"] == "val"]["pair_id"].tolist())
    (args.splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (args.splits_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    print("Wrote pairs:", args.out_csv)
    print("Wrote splits:", args.splits_dir)
    print("pairs:", len(out), "train:", len(train_ids), "val:", len(val_ids))
    print("skipped missing warped:", miss_warp)
    print("skipped missing cur mask_b:", miss_cur)
    print("skipped missing fallback prev mask_a:", miss_prev)
    print("example row:\n", out.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
