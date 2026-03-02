#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from maskscomp.change_detection import iter_tiles_2d, read_mask, read_pairs_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tile-wise change detection metrics from heatmaps")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--heatmap-dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def main() -> None:
    args = parse_args()
    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]
    all_scores: List[float] = []
    all_labels: List[int] = []
    hit_count = 0
    recall_num = 0
    recall_den = 0

    for r in rows:
        heat_path = args.heatmap_dir / r.sample_id / f"{Path(r.cur_path).stem}.npy"
        if not heat_path.exists():
            continue
        heat = np.load(heat_path)
        prev = read_mask(args.data_root / r.prev_path)
        cur = read_mask(args.data_root / r.cur_path)
        diff = (cur != prev).astype(np.uint8)

        labels = []
        for _y, _x, tile in iter_tiles_2d(diff, args.tile_size, args.stride):
            labels.append(1 if tile.mean() >= args.tau else 0)
        labels = np.asarray(labels, dtype=np.int64)
        scores = heat.reshape(-1).astype(np.float64)
        if scores.size != labels.size:
            n = min(scores.size, labels.size)
            scores = scores[:n]
            labels = labels[:n]

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

        k = min(args.topk, scores.size)
        top_idx = np.argsort(scores)[::-1][:k]
        hit_count += int(labels[top_idx].any()) if k > 0 else 0
        recall_num += int(labels[top_idx].sum()) if k > 0 else 0
        recall_den += int(labels.sum())

    y = np.asarray(all_labels, dtype=np.int64)
    s = np.asarray(all_scores, dtype=np.float64)
    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "method": args.method,
        "tau": args.tau,
        "ROC-AUC": _safe_auc(y, s),
        "PR-AUC": _safe_ap(y, s),
        "Hit@K": float(hit_count) / max(1, len(rows)),
        "Recall@K": float(recall_num) / max(1, recall_den),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        wr.writeheader()
        wr.writerow(metrics)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
