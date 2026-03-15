#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("/home/sasha/MasksComp/output/facades_pairs/facades_pairs_gtwarped_gtcur.csv"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/sasha/MasksComp/results_facades_conditional/all"),
        help="Will write: <out_root>/<pair_id>/residual_{C,V}/<stem>.png",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def imread_u8(p: Path):
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    # force uint8
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    return m


def main():
    args = parse_args()
    df = pd.read_csv(args.pairs_csv)
    need = ["pair_id", "prev_path", "cur_path"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"pairs csv missing columns: {miss}. columns={list(df.columns)}")

    if args.max_items is not None:
        df = df.head(int(args.max_items)).copy()

    ok = 0
    skip = 0

    frac_changed = []
    frac_v_zero = []

    for _, r in df.iterrows():
        pid = str(r["pair_id"])
        prevp = Path(str(r["prev_path"]))
        curp = Path(str(r["cur_path"]))

        if not prevp.exists() or not curp.exists():
            skip += 1
            continue

        prev = imread_u8(prevp)
        cur = imread_u8(curp)
        if prev is None or cur is None:
            skip += 1
            continue
        if prev.shape != cur.shape:
            skip += 1
            continue

        stem = Path(curp).stem  # e.g., Antonenko8_2025

        dC = args.out_root / pid / "residual_C"
        dV = args.out_root / pid / "residual_V"
        dC.mkdir(parents=True, exist_ok=True)
        dV.mkdir(parents=True, exist_ok=True)

        outC = dC / f"{stem}.png"
        outV = dV / f"{stem}.png"
        if (not args.overwrite) and (outC.exists() and outV.exists()):
            ok += 1
            continue

        C = (cur != prev).astype(np.uint8)           # 0/1
        V = (cur.astype(np.uint16) * C.astype(np.uint16)).astype(np.uint8)  # cur where changed else 0

        cv2.imwrite(str(outC), C)
        cv2.imwrite(str(outV), V)

        frac_changed.append(float((C != 0).mean()))
        frac_v_zero.append(float((V == 0).mean()))
        ok += 1

    print("rebuild ok:", ok, "skip:", skip, "total:", len(df))
    if frac_changed:
        a = np.asarray(frac_changed, dtype=np.float64)
        z = np.asarray(frac_v_zero, dtype=np.float64)
        print("C!=0 mean quantiles:", np.quantile(a, [0.0, 0.5, 0.9, 0.95, 1.0]))
        print("V==0 mean quantiles:", np.quantile(z, [0.0, 0.5, 0.9, 0.95, 1.0]))


if __name__ == "__main__":
    main()
