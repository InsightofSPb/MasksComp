#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np

from maskscomp.change_detection import read_pairs_csv


def parse_args():
    p = argparse.ArgumentParser("tile_agreement")
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--heat-a", type=Path, required=True, help="Dir with <pair_id>/<stem>.npy")
    p.add_argument("--heat-b", type=Path, required=True, help="Dir with <pair_id>/<stem>.npy")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--ids-txt", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)

    # NEW: disagreement tiles
    p.add_argument("--diff-topn", type=int, default=0, help="Save top-N most disagreeing tiles per pair (0 disables).")
    p.add_argument("--diff-metric", choices=["rank", "abs"], default="rank",
                  help="rank: |rankA-rankB|; abs: |normA-normB| with per-map minmax.")
    return p.parse_args()


def load_ids(p: Path) -> set[str]:
    s: set[str] = set()
    for ln in p.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if t:
            s.add(t)
    return s


def topk_set(flat: np.ndarray, k: int) -> set[int]:
    if flat.size == 0 or k <= 0:
        return set()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return set(int(i) for i in idx.tolist())


def summarize(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


def top_diff_tiles(flat_a: np.ndarray, flat_b: np.ndarray, n: int, metric: str) -> tuple[list[int], list[float]]:
    n = int(n)
    if n <= 0 or flat_a.size == 0 or flat_b.size == 0:
        return [], []
    m = min(flat_a.size, flat_b.size)
    a = flat_a[:m].astype(np.float64, copy=False)
    b = flat_b[:m].astype(np.float64, copy=False)

    if metric == "abs":
        da = _norm01(a)
        db = _norm01(b)
        diff = np.abs(da - db)
    else:
        # rank disagreement (0 = most salient). Larger -> bigger disagreement.
        order_a = np.argsort(-a)
        order_b = np.argsort(-b)
        rank_a = np.empty_like(order_a)
        rank_b = np.empty_like(order_b)
        rank_a[order_a] = np.arange(m)
        rank_b[order_b] = np.arange(m)
        diff = np.abs(rank_a - rank_b).astype(np.float64)

    n = min(n, m)
    idx = np.argpartition(diff, -n)[-n:]
    idx = idx[np.argsort(diff[idx])[::-1]]
    return [int(i) for i in idx.tolist()], [float(diff[i]) for i in idx.tolist()]


def main():
    args = parse_args()

    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]
    if args.ids_txt is not None:
        keep = load_ids(args.ids_txt)
        rows = [r for r in rows if str(r.pair_id) in keep]

    per = []
    j_list = []
    o_list = []

    for r in rows:
        stem = Path(r.cur_path).stem
        pa = args.heat_a / r.pair_id / f"{stem}.npy"
        pb = args.heat_b / r.pair_id / f"{stem}.npy"
        if not pa.exists() or not pb.exists():
            continue

        ha = np.load(pa).reshape(-1).astype(np.float64)
        hb = np.load(pb).reshape(-1).astype(np.float64)

        A = topk_set(ha, args.topk)
        B = topk_set(hb, args.topk)
        if len(A) == 0 and len(B) == 0:
            continue

        inter = len(A & B)
        union = len(A | B)
        jacc = inter / union if union > 0 else float("nan")
        overlap = inter / float(args.topk) if args.topk > 0 else float("nan")

        diff_idxs, diff_vals = top_diff_tiles(ha, hb, args.diff_topn, args.diff_metric) if args.diff_topn > 0 else ([], [])

        per.append((
            r.pair_id, stem,
            float(jacc), float(overlap),
            int(inter), int(union),
            int(args.topk),
            args.diff_metric, int(args.diff_topn),
            ";".join(map(str, diff_idxs)),
            ";".join(f"{v:.6g}" for v in diff_vals),
        ))

        j_list.append(float(jacc))
        o_list.append(float(overlap))

    j = np.array(j_list, dtype=np.float64)
    o = np.array(o_list, dtype=np.float64)

    print("pairs_evaluated:", len(per))
    print("Jaccard:", summarize(j))
    print("Overlap@K (|A∩B|/K):", summarize(o))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                "pair_id", "stem",
                "jaccard", "overlap_at_k", "inter", "union", "topk",
                "diff_metric", "diff_topn", "diff_tile_idxs", "diff_values"
            ])
            for row in per:
                wr.writerow(list(row))
        print("[OK] wrote", args.out_csv)


if __name__ == "__main__":
    main()
